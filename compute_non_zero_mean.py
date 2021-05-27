#!/usr/bin/env python
# -*- coding: utf-8

# For usage, type: python compute_non_zero_mean.py -h
#
# Authors: Sandrine Bédard

import argparse
import numpy as np
import nibabel as nib

def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes average accross 4th dimension excluding zero values" )
    parser.add_argument('-i', required=True, type=str,
                        help="Input image.")
    parser.add_argument('-o', required=True, type=str,
                        help="Ouput image.")

    return parser



def main():
    parser = get_parser()
    args = parser.parse_args()
    
    image = nib.load(args.i)
    image_np = np.array(image.dataobj)
    non_zeros = np.nonzero(image_np)
    print(non_zeros)
    
    # get all non-zeros
    # find min and max in 4rth dimension
    #



if __name__ == '__main__':
    main()