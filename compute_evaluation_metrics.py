#!/usr/bin/env python
# -*- coding: utf-8
# Computes mean dice coefficient across manual segmentations from candidates and ground truth segmentations. 
#
# For usage, type: python compute_dice.py -h
#
# Authors: Sandrine Bédard

import argparse
import logging
import os
import sys
import numpy as np
import nibabel as nib
import csv
from scipy import spatial


# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes dice coefficient between manual segmentations and ground truth segmentations.",
        prog=os.path.basename(__file__).strip('.py')
        )
    parser.add_argument('-im1',
                        required=True,
                        type=str,
                        metavar='<filename>',
                        help="Filename of image 1")
    parser.add_argument('-im2',
                        required=True,
                        type=str,
                        metavar='<filename>',
                        help="Filename of image 2")
    parser.add_argument('-o',
                        required=False,
                        type=str,
                        default='metrics.csv',
                        metavar='<filename>',
                        help="Output csv file.")
    return parser


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


def get_vol(data):
    """Get volume."""
    vol = np.sum(data.get_fdata())
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
    vol_gt = get_vol(im2)

    if vol_gt == 0.0:
        return np.nan

    rvd = (vol_gt - vol_pred)
    rvd /= vol_gt

    return rvd

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
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = (im1 * im2).sum()
    return (2. * intersection) / im_sum


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Initialize empty DataFrame
    fname_out = args.o
    im1 = nib.load(args.im1)
    im2 = nib.load(args.im2)
    dice = compute_dice(im1=im1.get_fdata(), im2=im2.get_fdata())
    precision = precision_score(im1.get_fdata(), im2.get_fdata())
    accuracy = accuracy_score(im1.get_fdata(), im2.get_fdata())
    specificity = specificity_score(im1.get_fdata(), im2.get_fdata())
    recall = recall_score(im1.get_fdata(), im2.get_fdata())
    mse_result = mse(im1.get_fdata(), im2.get_fdata())
    hausdorff = hausdorff_score(im1.get_fdata(), im2.get_fdata())
    

    # Create .csv file of results
    if not os.path.isfile(fname_out):
        with open(fname_out, 'w') as csvfile:
            header = ['Filename', 'Dice', 'Accuracy', 'Specificity', 'Recall', 'Precision', 'MSE', 'Hausdorff']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
    with open(fname_out, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        line = [args.im1, dice, accuracy, specificity, recall, precision, mse_result, hausdorff]
        spamwriter.writerow(line)


if __name__ == '__main__':
    main()
