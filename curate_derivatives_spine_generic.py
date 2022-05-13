#!/usr/bin/env python
# -*- coding: utf-8

# For usage, type: python curate_derivatives_spine_generic.py -h
#
# Authors: Sandrine Bédard

import argparse
import glob
import os
import shutil
import yaml
import json
import time


FOLDER_DERIVATIVES = os.path.join('derivatives', 'labels')


def get_parser():
    parser = argparse.ArgumentParser(
        description="Gets derivatives with label _seg-manual, removes suffix _RPI_r or _rms, creates json sidecar if it does not exist and creates a new curated derivatives folder. Ouptuts a .yml list with all manually corrected files.")
    parser.add_argument('-path-in', required=True, type=str,
                        help="Input path to derivatives folder to curate. Example: /derivatives/labels/")
    parser.add_argument('-path-out', required=True, type=str,
                        help="Output curated derivatives folder.")
    return parser


def create_json(fname_nifti, name_rater):
    """
    Create json sidecar with meta information
    :param fname_nifti: str: File name of the nifti image to associate with the json sidecar
    :param name_rater: str: Name of the expert rater
    :return:
    """
    metadata = {'Author': name_rater, 'Date': time.strftime('%Y-%m-%d %H:%M:%S')}
    fname_json = fname_nifti.rstrip('.nii').rstrip('.nii.gz') + '.json'
    with open(fname_json, 'w') as outfile:
        json.dump(metadata, outfile, indent=4)


def get_contrast(file):
    """
    Get contrast from BIDS file name
    :param file:
    :return:
    """
    if 'dwi' in file.split('_'):
        return 'dwi'
    else:
        return 'anat'


def splitext(fname):
    """
    Split a fname (folder/file + ext) into a folder/file and extension.

    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    """
    dir, filename = os.path.split(fname)
    for special_ext in ['.nii.gz', '.tar.gz']:
        if filename.endswith(special_ext):
            stem, ext = filename[:-len(special_ext)], special_ext
            return os.path.join(dir, stem), ext
    # If no special case, behaves like the regular splitext
    stem, ext = os.path.splitext(filename)
    return os.path.join(dir, stem), ext


def remove_suffix(fname, suffix):
    """
    Remove suffix between end of file name and extension.

    :param fname: absolute or relative file name with suffix. Example: t2_mean.nii
    :param suffix: suffix. Example: _mean
    :return: file name without suffix. Example: t2.nii

    Examples:

    - remove_suffix(t2_mean.nii, _mean) -> t2.nii
    - remove_suffix(t2a.nii.gz, a) -> t2.nii.gz
    """
    stem, ext = splitext(fname)
    return os.path.join(stem.replace(suffix, '') + ext)


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension.
    :param fname: absolute or relative file name. Example: t2.nii
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii
    Examples:
    - add_suffix(t2.nii, _mean) -> t2_mean.nii
    - add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """

    def _splitext(fname):
        """
        Split a fname (folder/file + ext) into a folder/file and extension.
        Note: for .nii.gz the extension is understandably .nii.gz, not .gz
        (``os.path.splitext()`` would want to do the latter, hence the special case).
        """
        dir, filename = os.path.split(fname)
        for special_ext in ['.nii.gz', '.tar.gz']:
            if filename.endswith(special_ext):
                stem, ext = filename[:-len(special_ext)], special_ext
                return os.path.join(dir, stem), ext
        # If no special case, behaves like the regular splitext
        stem, ext = os.path.splitext(filename)
        return os.path.join(dir, stem), ext

    stem, ext = _splitext(fname)
    return os.path.join(stem + suffix + ext)


def check_output_folder(path_bids, folder_derivatives):
    """
    Make sure path exists, has writing permissions, and create derivatives folder if it does not exist.
    :param path_bids:
    :return: path_bids_derivatives
    """
    if path_bids is None:
        print("-path-out should be provided.")
    if not os.path.exists(path_bids):
        print("Output path does not exist: {}".format(path_bids))
    path_bids_derivatives = os.path.join(path_bids, folder_derivatives)
    os.makedirs(path_bids_derivatives, exist_ok=True)
    return path_bids_derivatives


def get_subject(file):
    """
    Get subject from BIDS file name
    :param file:
    :return: subject
    """
    return file.split('_')[0]


def curate_csgseg(fname):
    """
    Reorient to RPI and resample _csg_seg-manual.nii.gz images.
    :param fname: absolute or relative file to curate.
    :return:
    """
    os.system('sct_image -i {} -setorient RPI -o {}'.format(fname, add_suffix(fname, '_RPI')))
    os.system('sct_resample -i {} -mm 0.8x0.8x0.8 -o {}'.format(add_suffix(fname, '_RPI'), add_suffix(fname, '_RPI_r')))

    try:
        os.remove(fname)
        os.remove(add_suffix(fname, '_RPI'))
    except OSError as e:  # if failed, report it back to the user ##
        print("Error: %s - %s." % (e.filename, e.strerror))

    shutil.move(add_suffix(fname, '_RPI_r'), fname)


def main():
    parser = get_parser()
    args = parser.parse_args()

    # check that output folder exists and has write permission
    path_out_deriv = check_output_folder(args.path_out, FOLDER_DERIVATIVES)

    name_rater = input("Enter your name (Firstname Lastname). It will be used to generate a json sidecar with each "
                       "corrected file: ")

    path_list = glob.glob(args.path_in + "/**/*seg-manual.nii.gz", recursive=True)  # TODO: add other extension
    # Get only filenames without absolute path
    file_list = [os.path.split(path)[-1] for path in path_list]

    # Initialize empty dict to create a list of all corrected files.
    manual_correction_list = {'FILES_SEG': []}

    for file in file_list:
        # build file names
        subject = get_subject(file)
        contrast = get_contrast(file)
        if contrast == 'dwi':
            file_curated = subject + '_rec-average_dwi_seg-manual.nii.gz'  # Rename
        else:
            file_curated = remove_suffix(file, '_RPI_r')  # Remove suffix on T1w and T2w images
            file_curated = remove_suffix(file_curated, '_rms')  # Remove suffix on T2star images
        # Append file to list.
        manual_correction_list['FILES_SEG'].append(remove_suffix(file_curated, '_seg-manual'))
        fname = os.path.join(args.path_in, subject, contrast, file)
        fname_label = os.path.join(
            path_out_deriv, subject, contrast, file_curated)
        os.makedirs(os.path.join(path_out_deriv, subject, contrast), exist_ok=True)
        shutil.copy(fname, fname_label)
    
        # Reorient and resample csfseg
        if 'csfseg-manual.nii.gz' in fname_label.split('_'):
            curate_csgseg(fname_label)

        # create json sidecar with the name of the expert rater if it doesn't exist.
        fname_json = fname.rstrip('.nii').rstrip('.nii.gz') + '.json'
        if not os.path.isfile(fname_json):
            create_json(fname_label, name_rater)
        else:
            fname_json_curated = fname_label.rstrip('.nii').rstrip('.nii.gz') + '.json'
            shutil.copy(fname_json, fname_json_curated)
    # Create a yml list with the name of the derivatives.
    with open(os.path.join(args.path_out, 'manual_seg.yml'), 'w') as file:
        yaml.dump(manual_correction_list, file)


if __name__ == '__main__':
    main()
