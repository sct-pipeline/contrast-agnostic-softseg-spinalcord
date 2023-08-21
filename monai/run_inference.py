import os
import argparse
import numpy as np
from loguru import logger
import torch.nn.functional as F
import torch
import json
from time import time

from monai.inferers import sliding_window_inference
from monai.data import (DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch)
from monai.transforms import (Compose, EnsureTyped, Invertd, SaveImage)

from transforms import val_transforms
from utils import precision_score, recall_score, dice_score
from models import ModifiedUNet3D

DEBUG = False
INIT_FILTERS=8
INFERENCE_ROI_SIZE = (160, 224, 96)   # (80, 192, 160)
DEVICE = "cpu"


def get_parser():

    parser = argparse.ArgumentParser(description="Run inference on a MONAI-trained model")

    parser.add_argument("--path-json", type=str, required=True, 
                        help="Path to the json file containing the test dataset in MSD format")
    parser.add_argument("--chkp-path", type=str, required=True, help="Path to the checkpoint folder")
    parser.add_argument("--path-out", type=str, required=True, 
                        help="Path to the output folder where to store the predictions and associated metrics")
    parser.add_argument("-dname", "--dataset-name", type=str, default="spine-generic",
                        help="Name of the dataset to run inference on")
    
    return parser


# --------------------------------
# DATA
# --------------------------------
def prepare_data(root, dataset_name="spine-generic"):
    # set deterministic training for reproducibility
    # set_determinism(seed=self.args.seed)
            
    # load the dataset
    dataset = os.path.join(root, f"{dataset_name}_dataset.json")
    # dataset = os.path.join(root, f"dataset_ivado_comparison.json")
    test_files = load_decathlon_datalist(dataset, True, "test")

    if DEBUG: # args.debug:
        test_files = test_files[:3]
    
    # define test transforms
    transforms_test = val_transforms(lbl_key='label')
    
    # define post-processing transforms for testing; taken (with explanations) from 
    # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
    test_post_pred = Compose([
        EnsureTyped(keys=["pred", "label"]),
        Invertd(keys=["pred", "label"], transform=transforms_test, 
                orig_keys=["image", "label"], 
                meta_keys=["pred_meta_dict", "label_meta_dict"],
                nearest_interp=False, to_tensor=True),
        ])
    test_ds = CacheDataset(data=test_files, transform=transforms_test, cache_rate=0.1, num_workers=4)

    return test_ds, test_post_pred


def main(args):

    # define start time
    start = time()

    # define root path for finding datalists
    dataset_root = args.path_json
    dataset_name = args.dataset_name

    chkp_path = os.path.join(args.chkp_path, "best_model.ckpt")

    results_path = args.path_out
    model_name = chkp_path.split("/")[-2]
    results_path = os.path.join(results_path, dataset_name, model_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    checkpoint = torch.load(chkp_path, map_location=torch.device(DEVICE))["state_dict"]
    # NOTE: remove the 'net.' prefix from the keys because of how the model was initialized in lightning
    # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
    for key in list(checkpoint.keys()):
        if 'net.' in key:
            checkpoint[key.replace('net.', '')] = checkpoint[key]
            del checkpoint[key]
    
    # initialize ivadomed unet model
    net = ModifiedUNet3D(in_channels=1, out_channels=1, init_filters=INIT_FILTERS)

    # load the trained model weights
    net.load_state_dict(checkpoint)
    net.to(DEVICE)

    # define the dataset and dataloader
    test_ds, test_post_pred = prepare_data(dataset_root, dataset_name)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # define list to collect the test metrics
    test_step_outputs = []
    test_summary = {}

    # iterate over the dataset and compute metrics
    net.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # compute time for inference per subject
            start_time = time()
            
            test_input = batch["image"]
            batch["pred"] = sliding_window_inference(test_input, INFERENCE_ROI_SIZE,
                                                    sw_batch_size=4, predictor=net, overlap=0.5)
            # NOTE: monai's models do not normalize the output, so we need to do it manually
            if bool(F.relu(batch["pred"]).max()):
                batch["pred"] = F.relu(batch["pred"]) / F.relu(batch["pred"]).max() 
            else:
                batch["pred"] = F.relu(batch["pred"])

            # # upon fsleyes visualization, observed that very small values need to be set to zero, but NOT fully binarizing the pred
            # batch["pred"][batch["pred"] < 0.099] = 0.0

            post_test_out = [test_post_pred(i) for i in decollate_batch(batch)]

            # make sure that the shapes of prediction and GT label are the same
            # print(f"pred shape: {post_test_out[0]['pred'].shape}, label shape: {post_test_out[0]['label'].shape}")
            assert post_test_out[0]['pred'].shape == post_test_out[0]['label'].shape
            
            pred, label = post_test_out[0]['pred'].cpu(), post_test_out[0]['label'].cpu()

            # save the prediction and label
            subject_name = (batch["image_meta_dict"]["filename_or_obj"][0]).split("/")[-1].replace(".nii.gz", "")
            print(f"Saving subject: {subject_name}")

            # image saver class
            save_folder = os.path.join(results_path, subject_name.split("_")[0])
            pred_saver = SaveImage(
                output_dir=save_folder, output_postfix="pred", output_ext=".nii.gz", 
                separate_folder=False, print_log=False)
            # save the prediction
            pred_saver(pred)

            # label_saver = SaveImage(
            #     output_dir=save_folder, output_postfix="gt", output_ext=".nii.gz", 
            #     separate_folder=False, print_log=False)
            # # save the label
            # label_saver(label)
                
            # NOTE: Important point from the SoftSeg paper - binarize predictions before computing metrics
            # calculate all metrics here
            # 1. Dice Score
            test_soft_dice = dice_score(pred, label)

            # binarizing the predictions 
            pred = (post_test_out[0]['pred'].detach().cpu() > 0.5).float()
            label = (post_test_out[0]['label'].detach().cpu() > 0.5).float()

            # 1.1 Hard Dice Score
            test_hard_dice = dice_score(pred.numpy(), label.numpy())
            # 2. Precision Score
            test_precision = precision_score(pred.numpy(), label.numpy())
            # 3. Recall Score
            test_recall = recall_score(pred.numpy(), label.numpy())

            end_time = time()
            metrics_dict = {
                "subject_name_and_contrast": subject_name,
                "dice_binary": round(test_hard_dice, 2),
                "dice_soft": round(test_soft_dice.item(), 2),
                "precision": round(test_precision, 2),
                "recall": round(test_recall, 2),
                # TODO: add relative volume difference here
                # NOTE: RVD is usually compared with binary objects (not soft)
                "inference_time_in_sec": round((end_time - start_time), 2),
            }
            test_step_outputs.append(metrics_dict)

        # save the test summary
        test_summary["metrics_per_subject"] = test_step_outputs

        # compute the average of all metrics
        avg_hard_dice_test, std_hard_dice_test = np.stack([x["dice_binary"] for x in test_step_outputs]).mean(), \
                                                    np.stack([x["dice_binary"] for x in test_step_outputs]).std()
        avg_soft_dice_test, std_soft_dice_test = np.stack([x["dice_soft"] for x in test_step_outputs]).mean(), \
                                                    np.stack([x["dice_soft"] for x in test_step_outputs]).std()
        avg_precision_test = np.stack([x["precision"] for x in test_step_outputs]).mean()
        avg_recall_test = np.stack([x["recall"] for x in test_step_outputs]).mean()
        avg_inference_time = np.stack([x["inference_time_in_sec"] for x in test_step_outputs]).mean()

        # store the average metrics in a dict
        avg_metrics = {
            "avg_dice_binary": round(avg_hard_dice_test, 2),
            "avg_dice_soft": round(avg_soft_dice_test, 2),
            "avg_precision": round(avg_precision_test, 2),
            "avg_recall": round(avg_recall_test, 2),
            "avg_inference_time_in_sec": round(avg_inference_time, 2),
        }
        test_summary["metrics_avg_across_cohort"] = avg_metrics

        logger.info(f"Test (Soft) Dice: {avg_soft_dice_test}")
        logger.info(f"Test (Hard) Dice: {avg_hard_dice_test}")
        logger.info(f"Test Precision Score: {avg_precision_test}")
        logger.info(f"Test Recall Score: {avg_recall_test}")
        logger.info(f"Average Inference Time per Subject: {avg_inference_time:.2f}s")
        
        # dump the test summary to a json file
        with open(os.path.join(results_path, "test_summary.json"), "w") as f:
            json.dump(test_summary, f, indent=4, sort_keys=True)

        # free up memory
        test_step_outputs.clear()

    end = time()

    print("=====================================================================")
    print(f"Total time taken for inference: {(end - start) / 60:.2f} minutes")
    print("=====================================================================")


if __name__ == "__main__":

    args = get_parser().parse_args()
    main(args)