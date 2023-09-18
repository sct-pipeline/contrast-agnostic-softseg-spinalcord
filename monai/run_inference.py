"""
Script to run inference on a MONAI-based model for contrast-agnostic soft segmentation of the spinal cord.
Predictions are stored in independent folders for each subject. Summary of the test metrics (both per subject and overall)
are stored in a json file, along with the time taken for inference.

Usage:
    python run_inference.py --path-json <path/to/json> --chkp-path <path/to/checkpoint> --path-out <path/to/output> 
                --model <unet/nnunet> --best-model-type <dice/loss> --crop_size <48x160x320> --device <gpu/cpu>

Author: Naga Karthik

"""

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

from transforms import val_transforms, val_transforms_with_orientation_and_crop
from utils import precision_score, recall_score, dice_score
from models import ModifiedUNet3D, create_nnunet_from_plans


# NNUNET global params
INIT_FILTERS=32
ENABLE_DS = True

nnunet_plans = {
    "UNet_class_name": "PlainConvUNet",
    "UNet_base_num_features": INIT_FILTERS,
    "n_conv_per_stage_encoder": [2, 2, 2, 2, 2, 2],
    "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
    "pool_op_kernel_sizes": [
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [1, 2, 2]
    ],
    "conv_kernel_sizes": [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3]
    ],
    "unet_max_num_features": 320,
}


def get_parser():

    parser = argparse.ArgumentParser(description="Run inference on a MONAI-trained model")

    parser.add_argument("--path-json", type=str, required=True, 
                        help="Path to the json file containing the test dataset in MSD format")
    parser.add_argument("--chkp-path", type=str, required=True, help="Path to the checkpoint folder")
    parser.add_argument("--path-out", type=str, required=True, 
                        help="Path to the output folder where to store the predictions and associated metrics")
    parser.add_argument("-dname", "--dataset-name", type=str, default="spine-generic",
                        help="Name of the dataset to run inference on")
    parser.add_argument("--model", type=str, default="unet", required=True,
                         help="Name of the model to use for inference")
    parser.add_argument("--best-model-type", type=str, default="dice", required=True, choices=["csa", "dice", "loss", "all"],
                            help="Type of the best model to use for inference i.e. based on csa/dice/both")
    # define args for cropping size. inputs should be in the format of "48x192x256"
    parser.add_argument('-crop', '--crop-size', type=str, default="48x160x320", 
                        help='Patch size used for center cropping the images during inference. Values correspond to R-L, A-P, I-S axes'
                        'of the image. Sliding window will be run across the cropped images. Use -1 if no cropping is intended '
                        '(sliding window will run across the whole image). Note, heavy R-L, A-P cropping is recommmended for best '
                        'results.  Default: 48x160x320')
    parser.add_argument('-debug', default=False, action='store_true', 
                        help='run inference only on a few images to check if things are working')
    parser.add_argument('--device', default="gpu", type=str, choices=["gpu", "cpu"],
                        help='Device to run inference on. Default: gpu')

    return parser


# --------------------------------
# DATA
# --------------------------------
def prepare_data(root, dataset_name="spine-generic", crop_size=(48, 160, 320)):
    # set deterministic training for reproducibility
    # set_determinism(seed=self.args.seed)
            
    # load the dataset
    dataset = os.path.join(root, f"{dataset_name}_dataset.json")
    test_files = load_decathlon_datalist(dataset, True, "test")

    if args.debug:
        test_files = test_files[:6]
    
    # define test transforms
    transforms_test = val_transforms_with_orientation_and_crop(crop_size=crop_size, lbl_key='label')
    
    # define post-processing transforms for testing; taken (with explanations) from 
    # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
    test_post_pred = Compose([
        EnsureTyped(keys=["pred", "label"]),
        Invertd(keys=["pred", "label"], transform=transforms_test, 
                orig_keys=["image", "label"], 
                meta_keys=["pred_meta_dict", "label_meta_dict"],
                nearest_interp=False, to_tensor=True),
        ])
    test_ds = CacheDataset(data=test_files, transform=transforms_test, cache_rate=0.25, num_workers=4)

    return test_ds, test_post_pred


def main(args):

    # define start time
    start = time()

    # define device
    if args.device == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU not available, using CPU instead")
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
    
    # define root path for finding datalists
    dataset_root = args.path_json
    dataset_name = args.dataset_name

    results_path = args.path_out
    model_name = args.chkp_path.split("/")[-1]
    if args.best_model_type == "dice":
        chkp_paths = [os.path.join(args.chkp_path, "best_model_dice.ckpt")]
        results_path = os.path.join(results_path, dataset_name, model_name, "best_dice")
    elif args.best_model_type == "loss":
        chkp_paths = [os.path.join(args.chkp_path, "best_model_loss.ckpt")]
        results_path = os.path.join(results_path, dataset_name, model_name)

    # save terminal outputs to a file
    logger.add(os.path.join(results_path, "logs.txt"), rotation="10 MB", level="INFO")

    logger.info(f"Saving results to: {results_path}")
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    # define cropping size
    inference_roi_size = tuple([int(i) for i in args.crop_size.split("x")])
    if inference_roi_size == (-1,):   # means no cropping is required
        logger.info(f"Doing Sliding Window Inference on Whole Images ...")
        inference_roi_size = (-1, -1, -1)

    # define the dataset and dataloader
    test_ds, test_post_pred = prepare_data(dataset_root, dataset_name, crop_size=inference_roi_size)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    if args.model == "unet":
        # initialize ivadomed unet model
        net = ModifiedUNet3D(in_channels=1, out_channels=1, init_filters=INIT_FILTERS)
    elif args.model == "nnunet":
        # define model
        net = create_nnunet_from_plans(plans=nnunet_plans, num_input_channels=1, num_classes=1, deep_supervision=ENABLE_DS)

    # define list to collect the test metrics
    test_step_outputs = []
    test_summary = {}
        
    preds_stack = []
    # iterate over the dataset and compute metrics
    with torch.no_grad():
        for batch in test_loader:
            # compute time for inference per subject
            start_time = time()

            # get the test input
            test_input = batch["image"].to(DEVICE)

            # load the checkpoints
            for chkp_path in chkp_paths:
            
                checkpoint = torch.load(chkp_path, map_location=torch.device(DEVICE))["state_dict"]
                # NOTE: remove the 'net.' prefix from the keys because of how the model was initialized in lightning
                # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
                for key in list(checkpoint.keys()):
                    if 'net.' in key:
                        checkpoint[key.replace('net.', '')] = checkpoint[key]
                        del checkpoint[key]
        
                # load the trained model weights
                net.load_state_dict(checkpoint)
                net.to(DEVICE)
                net.eval()

                # run inference            
                batch["pred"] = sliding_window_inference(test_input, inference_roi_size, mode="gaussian",
                                                        sw_batch_size=4, predictor=net, overlap=0.5, progress=False)

                if ENABLE_DS and args.model == "nnunet":
                    # take only the highest resolution prediction
                    batch["pred"] = batch["pred"][0]

                # NOTE: monai's models do not normalize the output, so we need to do it manually
                if bool(F.relu(batch["pred"]).max()):
                    batch["pred"] = F.relu(batch["pred"]) / F.relu(batch["pred"]).max() 
                else:
                    batch["pred"] = F.relu(batch["pred"])

                post_test_out = [test_post_pred(i) for i in decollate_batch(batch)]

                # make sure that the shapes of prediction and GT label are the same
                assert post_test_out[0]['pred'].shape == post_test_out[0]['label'].shape
                
                pred, label = post_test_out[0]['pred'].cpu(), post_test_out[0]['label'].cpu()
                
                # stack the predictions
                preds_stack.append(pred)

            # save the (soft) prediction and label
            subject_name = (batch["image_meta_dict"]["filename_or_obj"][0]).split("/")[-1].replace(".nii.gz", "")
            logger.info(f"Saving subject: {subject_name}")

            # take the average of the predictions
            pred = torch.stack(preds_stack).mean(dim=0)
            preds_stack.clear()

            # check whether the prediction and label have the same shape
            assert pred.shape == label.shape, f"Prediction and label shapes are different: {pred.shape} vs {label.shape}"

            # image saver class
            save_folder = os.path.join(results_path, subject_name.split("_")[0])
            pred_saver = SaveImage(
                output_dir=save_folder, output_postfix="pred", output_ext=".nii.gz", 
                separate_folder=False, print_log=False)
            # save the prediction
            pred_saver(pred)
                
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

    logger.info("=====================================================================")
    logger.info(f"Total time taken for inference: {(end - start) / 60:.2f} minutes")
    logger.info("=====================================================================")


if __name__ == "__main__":

    args = get_parser().parse_args()
    main(args)