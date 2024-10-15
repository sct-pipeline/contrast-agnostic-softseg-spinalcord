"""
Script to run inference on a MONAI-based model for contrast-agnostic soft segmentation of the spinal cord.

Author: Naga Karthik

"""

import os
import argparse
import numpy as np
import pydoc
import warnings
from loguru import logger
import torch.nn.functional as F
import torch
import torch.nn as nn
import json
from time import time
import yaml
from scipy import ndimage

from monai.inferers import sliding_window_inference
from monai.data import (DataLoader, Dataset, decollate_batch)
from monai.networks.nets import SwinUNETR
import monai.transforms as transforms

# ---------------------------- Imports for nnUNet's Model -----------------------------
from batchgenerators.utilities.file_and_folder_operations import join
from utils import recursive_find_python_class


nnunet_plans = {
    "arch_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
    "arch_kwargs": {
        "n_stages": 6,
        "features_per_stage": [32, 64, 128, 256, 384, 384],
        "strides": [
            [1, 1, 1], 
            [2, 2, 2], 
            [2, 2, 2], 
            [2, 2, 2],
            [2, 2, 2],
            [1, 2, 2]
        ],
        "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
        "n_conv_per_stage_decoder": [2, 2, 2, 2, 2]
    },
    "arch_kwargs_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
}

nnunet_plans_resencM = {
    "arch_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
    "arch_kwargs": {
        "n_stages": nnunet_plans["arch_kwargs"]["n_stages"],
        "features_per_stage": nnunet_plans["arch_kwargs"]["features_per_stage"],
        "strides": nnunet_plans["arch_kwargs"]["strides"],
        "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
        "n_conv_per_stage_decoder": [1, 1, 1, 1, 1]
    },
    "arch_kwargs_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
}

def get_parser():

    parser = argparse.ArgumentParser(description="Run inference on a MONAI-trained model")

    parser.add_argument("--path-img", type=str, required=True,
                        help="Path to the image to run inference on")
    parser.add_argument("--chkp-path", type=str, required=True, 
                        help="Path to the checkpoint folder. This folder should contain a file named 'best_model_loss.ckpt")
    parser.add_argument("--path-out", type=str, required=True, 
                        help="Path to the output folder where to store the predictions and associated metrics")
    parser.add_argument('-crop', '--crop-size', type=str, default="64x192x-1", 
                        help='Size of the window used to crop the volume before inference (NOTE: Images are resampled to 1mm'
                        ' isotropic before cropping). The window is centered in the middle of the volume. Dimensions are in the'
                        ' order R-L, A-P, I-S. Use -1 for no cropping in a specific axis, example: “64x160x-1”.'
                        ' NOTE: heavy R-L cropping is recommended for positioning the SC at the center of the image.'
                        ' Default: 64x192x-1')
    parser.add_argument('--device', default="gpu", type=str, choices=["gpu", "cpu"],
                        help='Device to run inference on. Default: cpu')
    parser.add_argument('--model', default="monai", type=str, choices=["monai", "monai-resencM", "swinunetr", "swinpretrained"], 
                        help='Model to use for inference. Default: monai')
    parser.add_argument('--pred-type', default="soft", type=str, choices=["soft", "hard"],
                        help='Type of prediction to output/save. `soft` outputs soft segmentation masks with a threshold of 0.1'
                        '`hard` outputs binarized masks thresholded at 0.5  Default: hard')
    parser.add_argument('--pad-mode', default="constant", type=str, choices=["constant", "edge", "reflect"],
                        help='Padding mode for the input image. Default: edge')
    parser.add_argument('--max-feat', default=384, type=int,
                        help='Maximum number of features in the network. Default: 320')
    return parser


# ===========================================================================
#                          Test-time Transforms
# ===========================================================================
def inference_transforms_single_image(crop_size, pad_mode="constant"):
    return transforms.Compose([
        transforms.LoadImaged(keys=["image"], image_only=False),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RPI"),
        transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=(2)),
        transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=crop_size, mode=pad_mode),
        # pad inputs to ensure divisibility by no. of layers nnUNet has (5)
        transforms.DivisiblePadd(keys=["image"], k=2**5, mode=pad_mode),
        transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])


# ============================================================================
#               Define the network based on nnunet_plans dict
# ============================================================================
def create_nnunet_from_plans(plans, input_channels, output_channels, allow_init = True, 
                             deep_supervision: bool = True):
    """
    Adapted from nnUNet's source code:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/get_network_from_plans.py#L9
    """

    network_class = plans["arch_class_name"]
    # only the keys that "could" depend on the dataset are defined in main.py
    architecture_kwargs = dict(**plans["arch_kwargs"])
    # rest of the default keys are defined here
    architecture_kwargs.update({
        "kernel_sizes": [
            [3, 3, 3], 
            [3, 3, 3], 
            [3, 3, 3], 
            [3, 3, 3], 
            [3, 3, 3], 
            [3, 3, 3],
        ],
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {
            "eps": 1e-05,
            "affine": True
        },
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"inplace": True},
    })

    for ri in plans["arch_kwargs_requires_import"]:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)
    # sometimes things move around, this makes it so that we can at least recover some of that
    if nw_class is None:
        warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                      f'dynamic_network_architectures.architectures...')
        
        import dynamic_network_architectures
        
        nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                               network_class.split(".")[-1],
                                               'dynamic_network_architectures.architectures')
        if nw_class is not None:
            print(f'FOUND IT: {nw_class}')
        else:
            raise ImportError('Network class could not be found, please check/correct your plans file')

    if deep_supervision is not None and 'deep_supervision' not in architecture_kwargs.keys():
        architecture_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network


# ===========================================================================
#                   Prepare temporary dataset for inference
# ===========================================================================
def prepare_data(path_image, crop_size=(64, 160, 320), pad_mode="edge"):

    test_file = [{"image": path_image}]
    
    # define test transforms
    transforms_test = inference_transforms_single_image(crop_size=crop_size, pad_mode=pad_mode)
    
    # define post-processing transforms for testing; taken (with explanations) from 
    # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
    test_post_pred = transforms.Compose([
        transforms.EnsureTyped(keys=["pred"]),
        transforms.Invertd(keys=["pred"], transform=transforms_test, 
                orig_keys=["image"], 
                meta_keys=["pred_meta_dict"],
                nearest_interp=False, to_tensor=True),
        ])
    test_ds = Dataset(data=test_file, transform=transforms_test)

    return test_ds, test_post_pred


# ===========================================================================
#                           Post-processing 
# ===========================================================================
def keep_largest_object(predictions):
    """Keep the largest connected object from the input array (2D or 3D).
    
    Taken from:
    https://github.com/ivadomed/ivadomed/blob/e101ebea632683d67deab3c50dd6b372207de2a9/ivadomed/postprocessing.py#L101-L116
    
    Args:
        predictions (ndarray or nibabel object): Input segmentation. Image could be 2D or 3D.

    Returns:
        ndarray or nibabel (same object as the input).
    """
    # Find number of closed objects using skimage "label"
    labeled_obj, num_obj = ndimage.label(np.copy(predictions))
    # If more than one object is found, keep the largest one
    if num_obj > 1:
        # Keep the largest object
        predictions[np.where(labeled_obj != (np.bincount(labeled_obj.flat)[1:].argmax() + 1))] = 0
    return predictions


# ===========================================================================
#                           Inference method
# ===========================================================================
def main():

    # get parameters
    args = get_parser().parse_args()

    # define device
    if args.device == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU not available, using CPU instead")
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
    
    # define root path for finding datalists
    path_image = args.path_img
    results_path = args.path_out
    chkp_path = os.path.join(args.chkp_path, "best_model.ckpt")

    # save terminal outputs to a file
    logger.add(os.path.join(results_path, "logs.txt"), rotation="10 MB", level="INFO")

    logger.info(f"Saving results to: {results_path}")
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    # define inference patch size and center crop size
    crop_size = tuple([int(i) for i in args.crop_size.split("x")])
    inference_roi_size = (64, 192, 320)

    # define the dataset and dataloader
    test_ds, test_post_pred = prepare_data(path_image, crop_size=crop_size, pad_mode=args.pad_mode)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # temporary fix for the nnUNet model because v2x was trained with 320 max features but the newer 
    # models have 384 max features
    nnunet_plans["arch_kwargs"]["features_per_stage"] = [32, 64, 128, 256, args.max_feat, args.max_feat]

    # define model
    if args.model == "monai":
        net = create_nnunet_from_plans(plans=nnunet_plans, input_channels=1, 
                                       output_channels=1, deep_supervision=True)
    elif args.model == "monai-resencM":
        net = create_nnunet_from_plans(plans=nnunet_plans_resencM, input_channels=1,
                                       output_channels=1, deep_supervision=True)    
    
    elif args.model in ["swinunetr", "swinpretrained"]:
        # load config file
        config_path = os.path.join(args.chkp_path, "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        net = SwinUNETR(
            spatial_dims=config["model"]["swinunetr"]["spatial_dims"],
            in_channels=1, out_channels=1, 
            img_size=config["preprocessing"]["crop_pad_size"],
            depths=config["model"]["swinunetr"]["depths"],
            feature_size=config["model"]["swinunetr"]["feature_size"], 
            num_heads=config["model"]["swinunetr"]["num_heads"])
        
    else:
        raise ValueError("Model not recognized. Please choose from: nnunet, swinunetr")


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

            if args.model in ["monai", "monai-resencM"]:
                # take only the highest resolution prediction
                # NOTE: both these models use Deep Supervision, so only the highest resolution prediction is taken
                batch["pred"] = batch["pred"][0]

            # NOTE: monai's models do not normalize the output, so we need to do it manually
            if bool(F.relu(batch["pred"]).max()):
                batch["pred"] = F.relu(batch["pred"]) / F.relu(batch["pred"]).max() 
            else:
                batch["pred"] = F.relu(batch["pred"])

            post_test_out = [test_post_pred(i) for i in decollate_batch(batch)]
            
            pred = post_test_out[0]['pred'].cpu()
            
            # threshold or binarize the output based on the pred_type
            if args.pred_type == "soft":
                pred[pred < 0.1] = 0
            elif args.pred_type == "hard":
                pred = torch.where(pred > 0.5, 1, 0)

            # keep the largest connected object
            pred = keep_largest_object(pred)

            # get subject name
            subject_name = (batch["image_meta_dict"]["filename_or_obj"][0]).split("/")[-1].replace(".nii.gz", "")
            logger.info(f"Saving subject: {subject_name}")

            # this takes about 0.25s on average on a CPU
            # image saver class
            pred_saver = transforms.SaveImage(
                output_dir=results_path, output_postfix="pred", output_ext=".nii.gz", 
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


if __name__ == "__main__":
    main()
