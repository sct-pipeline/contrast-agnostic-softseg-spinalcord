"""
Script to run inference on a MONAI-based model for contrast-agnostic soft segmentation of the spinal cord.
Prediction is stored in an independent folder given by subject name. The time taken for inference is stored in a json file.

Usage:
    python run_inference_single_image.py --path-img /path/to/my-awesome-SC-image.nii.gz --chkp-path /path/to/best/model 
                --path-out /path/to/output/folder --crop-size <64x160x320> --device <cpu/gpu>

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

from models import create_nnunet_from_plans
from monai.inferers import sliding_window_inference
from monai.data import (DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch)
from monai.transforms import (Compose, EnsureTyped, Invertd, SaveImage, Spacingd,
                              LoadImaged, NormalizeIntensityd, EnsureChannelFirstd, 
                              DivisiblePadd, Orientationd, ResizeWithPadOrCropd)


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

    parser.add_argument("--path-img", type=str, required=True,
                        help="Path to the image to run inference on")
    parser.add_argument("--chkp-path", type=str, required=True, help="Path to the checkpoint folder")
    parser.add_argument("--path-out", type=str, required=True, 
                        help="Path to the output folder where to store the predictions and associated metrics")
    parser.add_argument('-crop', '--crop-size', type=str, default="48x160x320", 
                        help='Patch size used for center cropping the images during inference. Values correspond to R-L, A-P, I-S axes'
                        'of the image. Sliding window will be run across the cropped images. Use -1 if no cropping is intended '
                        '(sliding window will run across the whole image). Note, heavy R-L, A-P cropping is recommmended for best '
                        'results.  Default: 48x160x320')
    parser.add_argument('--device', default="gpu", type=str, choices=["gpu", "cpu"],
                        help='Device to run inference on. Default: gpu')

    return parser


# define transforms for inference
def inference_transforms_single_image(crop_size):
    return Compose([
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RPI"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=(2)),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=crop_size,),
            DivisiblePadd(keys=["image"], k=2**5),   # pad inputs to ensure divisibility by no. of layers nnUNet has (5)
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])

# --------------------------------
# DATA
# --------------------------------
def prepare_data(path_image, path_out, crop_size=(48, 160, 320)):

    # create a temporary datalist containing the image
    # boiler plate keys to be defined in the MSD-style datalist
    params = {}
    params["description"] = "my-awesome-SC-image"
    params["labels"] = {
        "0": "background",
        "1": "soft-sc-seg"
        }
    params["modality"] = {
        "0": "MRI"
        }
    params["tensorImageSize"] = "3D"
    params["test"] = [
        {
            "image": path_image
        }
    ]

    final_json = json.dumps(params, indent=4, sort_keys=True)
    jsonFile = open(path_out + "/" + f"temp_msd_datalist.json", "w")
    jsonFile.write(final_json)
    jsonFile.close()    

    dataset = os.path.join(path_out, f"temp_msd_datalist.json")    
    test_files = load_decathlon_datalist(dataset, True, "test")
    
    # define test transforms
    transforms_test = inference_transforms_single_image(crop_size=crop_size)
    
    # define post-processing transforms for testing; taken (with explanations) from 
    # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
    test_post_pred = Compose([
        EnsureTyped(keys=["pred"]),
        Invertd(keys=["pred"], transform=transforms_test, 
                orig_keys=["image"], 
                meta_keys=["pred_meta_dict"],
                nearest_interp=False, to_tensor=True),
        ])
    test_ds = CacheDataset(data=test_files, transform=transforms_test, cache_rate=0.75, num_workers=8)

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
    path_image = args.path_img
    results_path = args.path_out
    chkp_path = os.path.join(args.chkp_path, "best_model_loss.ckpt")

    # save terminal outputs to a file
    logger.add(os.path.join(results_path, "logs.txt"), rotation="10 MB", level="INFO")

    logger.info(f"Saving results to: {results_path}")
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    # define inference patch size and center crop size
    crop_size = tuple([int(i) for i in args.crop_size.split("x")])
    inference_roi_size = (64, 160, 320)

    # define the dataset and dataloader
    test_ds, test_post_pred = prepare_data(path_image, results_path, crop_size=crop_size)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # define model
    net = create_nnunet_from_plans(plans=nnunet_plans, num_input_channels=1, num_classes=1, deep_supervision=ENABLE_DS)

    # define list to collect the test metrics
    test_step_outputs = []
    test_summary = {}
        
    # iterate over the dataset and compute metrics
    with torch.no_grad():
        for batch in test_loader:

            # compute time for inference per subject
            start_time = time()

            # get the test input
            test_input = batch["image"].to(DEVICE)
            
            # this loop only takes about 0.2s on average on a CPU
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

            # take only the highest resolution prediction
            batch["pred"] = batch["pred"][0]

            # NOTE: monai's models do not normalize the output, so we need to do it manually
            if bool(F.relu(batch["pred"]).max()):
                batch["pred"] = F.relu(batch["pred"]) / F.relu(batch["pred"]).max() 
            else:
                batch["pred"] = F.relu(batch["pred"])

            post_test_out = [test_post_pred(i) for i in decollate_batch(batch)]
            
            pred = post_test_out[0]['pred'].cpu()
            
            # clip the prediction between 0.5 and 1
            pred = torch.clamp(pred, 0.5, 1)
            # # threshold the prediction
            # pred = (pred > 0.1).float()
                
            # get subject name
            subject_name = (batch["image_meta_dict"]["filename_or_obj"][0]).split("/")[-1].replace(".nii.gz", "")
            logger.info(f"Saving subject: {subject_name}")

            # this takes about 0.25s on average on a CPU
            # image saver class
            save_folder = os.path.join(results_path, subject_name.split("_")[0])
            pred_saver = SaveImage(
                output_dir=save_folder, output_postfix="pred", output_ext=".nii.gz", 
                separate_folder=False, print_log=False)
            # save the prediction
            pred_saver(pred)
                
            end_time = time()
            metrics_dict = {
                "subject_name_and_contrast": subject_name,
                "inference_time_in_sec": round((end_time - start_time), 2),
            }
            test_step_outputs.append(metrics_dict)

        # save the test summary
        test_summary["metrics_per_subject"] = test_step_outputs

        # compute the average inference time
        avg_inference_time = np.stack([x["inference_time_in_sec"] for x in test_step_outputs]).mean()

        # store the average metrics in a dict
        avg_metrics = {
            "avg_inference_time_in_sec": round(avg_inference_time, 2),
        }
        test_summary["metrics_avg_across_cohort"] = avg_metrics

        logger.info("========================================================")
        logger.info(f"      Inference Time per Subject: {avg_inference_time:.2f}s")
        logger.info("========================================================")

        
        # dump the test summary to a json file
        with open(os.path.join(results_path, "test_summary.json"), "w") as f:
            json.dump(test_summary, f, indent=4, sort_keys=True)

        # free up memory
        test_step_outputs.clear()
        test_summary.clear()
        os.remove(os.path.join(results_path, "temp_msd_datalist.json"))

    end = time()

    # logger.info("===============================================================")
    # logger.info(f"Total time taken for inference: {(end - start) / 60:.2f} minutes")
    # logger.info("===============================================================")


if __name__ == "__main__":

    args = get_parser().parse_args()
    main(args)