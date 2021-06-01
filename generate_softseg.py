#!/usr/bin/env python
# -*- coding: utf-8

# For usage, type: python generate_softseg.py -h
#
# Authors: Sandrine Bédard

import argparse
import numpy as np
import nibabel as nib

def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes average accross 4th dimension excluding zero values | Orientation needs to be RPI" )
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
   # image_np = np.clip(image_np, 0, 1) # Remove values higher than 1 or do it after??
    images_FOV = np.zeros((image_np.shape[-1], 2), dtype=int)
    image_soft_seg = np.zeros(image_np.shape[-1])
    # Get indexes of max and min of segmentation for each images
    for i in range(image_np.shape[-1]):
        index_non_zero = np.argwhere(image_np[:,:,:,i]!=0)
        min_index = np.argmin(index_non_zero[:,2])
        min = index_non_zero[min_index][2]
        max_index = np.argmax(index_non_zero[:,2])
        max = index_non_zero[max_index][2]
        images_FOV[i] = [min,max]
    for i in range(image_np.shape[-1]):
        image_np_cp = image_np.copy()
        small_image_index = np.argmin(images_FOV[:,1]-images_FOV[:,0])
        low_FOV = images_FOV[small_image_index,0]
        high_FOV = images_FOV[small_image_index,1]
        image_np_cp[:,:,0:low_FOV,:] = 0
        image_np_cp[:,:,high_FOV+1::,:] = 0
        if i==0:
            image_soft_seg = np.mean(image_np_cp, axis=3)
        else:
            image_soft_seg[:,:,0:previous_min] = np.mean(image_np_cp, axis=3)[:,:,0:previous_min]
            image_soft_seg[:,:,previous_max::] = np.mean(image_np_cp, axis=3)[:,:,previous_max::]
        
        previous_min = low_FOV
        previous_max = high_FOV
        images_FOV =  np.delete(images_FOV, small_image_index, axis=0)
        image_np = np.delete(image_np, small_image_index, axis=3)
    image_soft_seg = np.clip(image_soft_seg, 0, 1)
    save_Nifti1(image_soft_seg, image, args.o)

if __name__ == '__main__':
    main()