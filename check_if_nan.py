#!/usr/bin/env python
# -*- coding: utf-8
# Add padding to an image by mirroring the image
#
# For usage, type: python pad_seg -h

# Authors: Sandrine Bédard


import argparse
import numpy as np
import nibabel as nib

def get_parser():
    parser = argparse.ArgumentParser(
        description="Add padding to a segmentation by 1 slice on both extremities| Orientation needs to be RPI" )
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
    idx_nan = np.argwhere(np.isnan(image_np))
    print(idx_nan)
    image_np = np.nan_to_num(image_np)
    idx_nan = np.argwhere(np.isnan(image_np))
    print(idx_nan)
    save_Nifti1(image_np, image, args.o)

    
if __name__ == '__main__':
    main() 