#!/usr/bin/env python
# -*- coding: utf-8
# Add padding to an image by mirroring the image
#
# For usage, type: python remove_slices_seg.py -h

# Authors: Sandrine Bédard

import argparse
import numpy as np
import nibabel as nib

NEAR_ZERO_THRESHOLD = 1e-6


def get_parser():
    parser = argparse.ArgumentParser(
        description="TODO| Orientation needs to be RPI" )
    parser.add_argument('-i', required=True, type=str,
                        help="Input segmentation in T2w space.")
    parser.add_argument('-coverage-map', required=True, type=str,
                        help="Coverage map in T2w space.")
    parser.add_argument('-c', required=True, type=str,
                        help="Contrast.")

    return parser


def save_Nifti1(data, original_image, filename):
    empty_header = nib.Nifti1Header()
    image = nib.Nifti1Image(data, original_image.affine, empty_header)
    nib.save(image, filename)


def remove_slices(image, max_z_index, min_z_index, nb_slices):
    image_seg_crop = image.copy()
    # Remove top slices
    image_seg_crop[:, :, (max_z_index - nb_slices)::] = 0
    image_seg_crop[:, :, 0:(min_z_index + nb_slices + 1)] = 0
    return image_seg_crop


def main():
    parser = get_parser()
    args = parser.parse_args()

    contrast_seg = args.c
    image = nib.load(args.i)
    image_np = image.get_fdata()
    coverage_map = nib.load(args.coverage_map)
    coverage_map_np = coverage_map.get_fdata()

    # Get max and min index of the segmentation
    _, _, Z = (image_np > NEAR_ZERO_THRESHOLD).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # Number of slices to remove:
    # MTS: 10 to and bottom
    # DWI & T2star : 8 slices
    if contrast_seg == "t2s" or contrast_seg == "dwi":
        nb_slices = 7
    else:
        nb_slices = 9
    image_np_crop = remove_slices(image_np, max_z_index, min_z_index, nb_slices)
    coverage_map_np_crop = remove_slices(coverage_map_np, max_z_index, min_z_index, nb_slices)

    # Find index of top and low slice
    # remove n slices of segmentation and coverage map
    # save new nifti for both
    filename_seg_crop = (args.i).split('.')[0] + "_crop" + ".nii.gz"
    filename_coverage_crop = (args.coverage_map).split('.')[0] + "_crop" + ".nii.gz"
    save_Nifti1(image_np_crop, image, filename_seg_crop)
    save_Nifti1(coverage_map_np_crop, coverage_map, filename_coverage_crop)


if __name__ == '__main__':
    main() 