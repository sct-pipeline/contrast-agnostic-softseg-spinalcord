#!/usr/bin/env python
# -*- coding: utf-8
# Computes mean dice coefficient across manual segmentations from candidates and ground truth segmentations. 
#
# For usage, type: python compute_dice.py -h
#
# Authors: Sandrine Bédard & Adrian El Baz

import argparse
import logging
import os
import sys
import numpy as np
import nibabel as nib
import csv
import pandas as pd
from scipy import spatial
from pathlib import Path

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:
    * FP = Soft False Positives
    * FN = Soft False Negatives
    * TP = Soft True Positives
    * TN = Soft True Negatives
    Robust to hard or soft input masks. For example::
        prediction=np.asarray([0, 0.5, 1])
        groundtruth=np.asarray([0, 1, 1])
        Leads to FP = 1.5
    Note: It assumes input values are between 0 and 1.
    Args:
        prediction (ndarray): Binary prediction.
        groundtruth (ndarray): Binary groundtruth.
    Returns:
        float, float, float, float: FP, FN, TP, TN
    """
    FP = float(np.sum(prediction * (1.0 - groundtruth)))
    FN = float(np.sum((1.0 - prediction) * groundtruth))
    TP = float(np.sum(prediction * groundtruth))
    TN = float(np.sum((1.0 - prediction) * (1.0 - groundtruth)))
    return FP, FN, TP, TN


def get_vol(data, remove_lastdim=False):
    """Get volume."""
    ## Binarize
    if remove_lastdim:
        vol = np.sum(np.squeeze(np.where(np.asarray(data.get_fdata())>0.5, 1, 0), axis=3))
    else: 
        vol = np.sum(np.where(np.asarray(data.get_fdata())>0.5, 1, 0))
    #vol = np.sum(data.get_fdata())
    px, py, pz = data.header['pixdim'][1:4]
    vol *= px * py * pz
    return vol


def precision_score(prediction, groundtruth, err_value=0.0):
    """Positive predictive value (PPV).
    Precision equals the number of true positive voxels divided by the sum of true and false positive voxels.
    True and false positives are computed on soft masks, see ``"numeric_score"``.
    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
        err_value (float): Value returned in case of error.
    Returns:
        float: Precision score.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP) <= 0.0:
        return err_value

    precision = np.divide(TP, TP + FP)
    return precision


def recall_score(prediction, groundtruth, err_value=0.0):
    """True positive rate (TPR).
    Recall equals the number of true positive voxels divided by the sum of true positive and false negative voxels.
    True positive and false negative values are computed on soft masks, see ``"numeric_score"``.
    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
        err_value (float): Value returned in case of error.
    Returns:
        float: Recall score.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return err_value
    TPR = np.divide(TP, TP + FN)
    return TPR


def specificity_score(prediction, groundtruth, err_value=0.0):
    """True negative rate (TNR).
    Specificity equals the number of true negative voxels divided by the sum of true negative and false positive voxels.
    True negative and false positive values are computed on soft masks, see ``"numeric_score"``.
    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
        err_value (float): Value returned in case of error.
    Returns:
        float: Specificity score.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TN + FP) <= 0.0:
        return err_value
    TNR = np.divide(TN, TN + FP)
    return TNR


def hausdorff_score(prediction, groundtruth):
    """Compute the directed Hausdorff distance between two N-D arrays.
    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
    Returns:
        float: Hausdorff distance.
    """
    if len(prediction.shape) == 4:
        n_classes, height, depth, width = prediction.shape
        # Reshape to have only 3 dimensions where prediction[:, idx, :] represents each 2D slice
        prediction = prediction.reshape((height, n_classes * depth, width))
        groundtruth = groundtruth.reshape((height, n_classes * depth, width))

    if len(prediction.shape) == 3:
        mean_hansdorff = 0
        for idx in range(prediction.shape[1]):
            pred = prediction[:, idx, :]
            gt = groundtruth[:, idx, :]
            mean_hansdorff += spatial.distance.directed_hausdorff(pred, gt)[0]
        mean_hansdorff = mean_hansdorff / prediction.shape[1]
        return mean_hansdorff

    return spatial.distance.directed_hausdorff(prediction, groundtruth)[0]


def accuracy_score(prediction, groundtruth, err_value=0.0):
    """Accuracy.
    Accuracy equals the number of true positive and true negative voxels divided by the total number of voxels.
    True positive/negative and false positive/negative values are computed on soft masks, see ``"numeric_score"``.
    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
    Returns:
        float: Accuracy.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    if N <= 0.0:
        return err_value
    accuracy = np.divide(TP + TN, N)
    return accuracy


def mse(im1, im2):
    """Compute the Mean Squared Error.
    Compute the Mean Squared Error between the two images, i.e. sum of the squared difference.
    Args:
        im1 (ndarray): First array.
        im2 (ndarray): Second array.
    Returns:
        float: Mean Squared Error.
    """
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    err = np.sum((im1.astype("float") - im2.astype("float")) ** 2)
    err /= float(im1.shape[0] * im1.shape[1])

    return err

def get_rvd(im1, im2):
    """Relative volume difference.
    The volume is here defined by the physical volume, in mm3, of the non-zero voxels of a given mask.
    Relative volume difference equals the difference between the ground-truth and prediction volumes, divided by the
    ground-truth volume.
    Optimal value is zero. Negative value indicates over-segmentation, while positive value indicates
    under-segmentation.
    """
    vol_pred = get_vol(im1)
    vol_gt = get_vol(im2, remove_lastdim=True) # Pred mask 4th dim removal

    if vol_gt == 0.0:
        return np.nan

    rvd = (vol_gt - vol_pred)
    rvd /= vol_gt

    return rvd


def get_avd(im1, im2):
    """Absolute volume difference.
    The volume is here defined by the physical volume, in mm3, of the non-zero voxels of a given mask.
    Absolute volume difference equals the absolute value of the Relative Volume Difference.
    Optimal value is zero.
    """
    return abs(get_rvd(im1, im2))


def compute_dice(im1, im2, empty_score=np.nan):
    """Computes the Dice coefficient between im1 and im2.
    Compute a soft Dice coefficient between im1 and im2, ie equals twice the sum of the two masks product, divided by
    the sum of each mask sum.
    If both images are empty, then it returns empty_score.
    Args:
        im1 (ndarray): First array.
        im2 (ndarray): Second array.
        empty_score (float): Returned value if both input array are empty.
    Returns:
        float: Dice coefficient.
    """
    im1 = np.asarray(im1)
    im2 = np.squeeze(np.asarray(im2), axis=3) # Fix pred mask 4th dim
    # Binarization threshold
    im1 = np.where(im1>0.5, 1, 0)
    im2 = np.where(im2>0.5, 1, 0)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = (im1 * im2).sum()
    return (2. * intersection) / im_sum

def compute_folder_metrics(base_folder: str, exclude_list: list = []):
    data_all_path = "../duke/temp/adrian/contrast-agnostic-seg-models/data_processed_clean"
    data_MTon_MTS = "../duke/temp/adrian/contrast-agnostic-seg-models/data_processed_clean_MTon_MTS"
    data_T1w_MTS = "../duke/temp/adrian/contrast-agnostic-seg-models/data_processed_clean_T1w_MTS"

    exp_folders = next(os.walk(base_folder))[1]
    if exclude_list:
        exp_folders = [f for f in exp_folders for excl in exclude_list if excl not in f]
    for exp in exp_folders:
        pred_mask_paths = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(os.path.join(base_folder, exp, "pred_masks")) for f in filenames if f.endswith("pred.nii.gz")]
        prefix_paths = [pth.split('/')[-1].split('_pred')[0] for pth in pred_mask_paths]
        patients = [pth.split('_')[0] for pth in prefix_paths]
        exp_result = {"Filename": [], "Dice": [], "RVD": []}
        for (patient, prefix, pred_mask_path) in zip(patients, prefix_paths, pred_mask_paths):
            if "flip-1" in prefix: # different MTS not currently recogizable
                data_folder = data_MTon_MTS
            elif "flip-2" in prefix: # different MTS not currently recogizable
                data_folder = data_T1w_MTS
            else:
                data_folder= data_all_path
            type_folder = "anat" if "dwi" not in prefix else "dwi"
            ground_truth_path = os.path.join(data_folder, f"derivatives/labels_softseg/{patient}/{type_folder}/{prefix}_softseg.nii.gz" )
            im1 = nib.load(ground_truth_path)
            im2 = nib.load(pred_mask_path)
            
            # Binarization occurs directly in metric computations
            dice = compute_dice(im1.get_fdata(), im2.get_fdata())
            rvd = get_rvd(im1,im2)
            exp_result["Filename"].append(pred_mask_path)
            exp_result["Dice"].append(dice)
            exp_result["RVD"].append(rvd)

        df = pd.DataFrame.from_dict(exp_result)
        if not os.path.isdir(f"./spine-generic-test-results/{exp}/"):
            Path(f"./spine-generic-test-results/{exp}/").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"./spine-generic-test-results/{exp}/evaluation_3Dmetrics.csv")

def main():
    """2 Experiment folders were used. One for MTS contrasts and "all", the other 
    for the rest of the experiments. Replace the folder names by your own experiment
    folders.
    """

    g9_base_folder = "../duke/temp/adrian/contrast-agnostic-seg-models/Group9_20-12-2022/"
    g8_base_folder = "../duke/temp/adrian/contrast-agnostic-seg-models/Group8_01-12-2022/"

    compute_folder_metrics(g9_base_folder)
    compute_folder_metrics(g8_base_folder, ["all"])

if __name__ == '__main__':
    main()