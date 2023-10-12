#!/usr/bin/env python
# -*- coding: utf-8
# Clip softsegs between 0 and 1.
#
# For usage, type: python clip_softseg.py -h

# Authors: Sandrine Bédard

import argparse
import numpy as np
import nibabel as nib


def get_parser():
    parser = argparse.ArgumentParser(
        description="Clip softsegs between 0 and 1." )
    parser.add_argument('-i', required=True, type=str,
                        help="Input image.")
    parser.add_argument('-o', required=True, type=str,
                        help="Ouput image.")

    return parser


def save_Nifti1(data, original_image, filename):
    empty_header = nib.Nifti1Header()
    image = nib.Nifti1Image(data, original_image.affine, empty_header)
    nib.save(image, filename)


def main():
    parser = get_parser()
    args = parser.parse_args()
    image = nib.load(args.i)
    image_np = image.get_fdata()
    image_np = np.clip(image_np, 0, 1)
    save_Nifti1(image_np, image, args.o)


if __name__ == '__main__':
    main()


