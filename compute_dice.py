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
                        default='dice.csv',
                        metavar='<filename>',
                        help="Output csv file.")
    return parser


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

    # Create .csv file of results
    if not os.path.isfile(fname_out):
        with open(fname_out, 'w') as csvfile:
            header = ['Filename', 'Dice']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
    with open(fname_out, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        line = [args.im1, dice]
        spamwriter.writerow(line)


if __name__ == '__main__':
    main()
