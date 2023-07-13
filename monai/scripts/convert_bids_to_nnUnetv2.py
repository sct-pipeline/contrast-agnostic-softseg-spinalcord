"""
Converts the BIDS-structured spine-generic dataset to the nnUNetv2 dataset format. Full details about 
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
Note that the conversion from BIDS to nnUNet is done using symbolic links to avoid creating multiple copies of the 
(original) BIDS dataset.
Currently only supports the conversion of a single contrast. In case of multiple contrasts, the script should be 
modified to include those as well. 
Usage example:
    python convert_spine-generic_to_nnUNetv2.py --path-data /path/to/spine-generic --path-out /path/to/nnunet 
        --dataset-name spineGen --dataset-number 701 --split 0.8 0.2
    
"""

import argparse
import pathlib
from pathlib import Path
import json
import os
from collections import OrderedDict
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

import nibabel as nib
import numpy as np


def binarize_label(subject_path, label_path):
    label_npy = nib.load(label_path).get_fdata()
    threshold = 1e-8
    label_npy = np.where(label_npy > threshold, 1, 0)
    ref = nib.load(subject_path)
    label_bin = nib.Nifti1Image(label_npy, ref.affine, ref.header)
    # overwrite the original label file with the binarized version
    nib.save(label_bin, label_path)


# parse command line arguments
parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
parser.add_argument('--path-data', help='Path to BIDS dataset.', required=True)
parser.add_argument('--path-out', help='Path to output directory.', required=True)
parser.add_argument('--dataset-name', '-dname', default='MSSpineLesion', type=str,
                    help='Specify the task name - usually the anatomy to be segmented, e.g. Hippocampus',)
parser.add_argument('--label-suffix', default='seg-manual', type=str,
                    help='Specify name of the segmentation to use.',)
parser.add_argument('--dataset-number', '-dnum', default=501,type=int, 
                    help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')
parser.add_argument("--contrasts", required=True, nargs="*", help="Contrasts to build our dataset from.")
parser.add_argument('--seed', default=42, type=int, 
                    help='Seed to be used for the random number generator split into training and test sets.')
# argument that accepts a list of floats as train val test splits
parser.add_argument('--split', nargs='+', required=True, type=float, default=[0.8, 0.2],
                    help='Ratios of training (includes validation) and test splits lying between 0-1. Example: --split 0.8 0.2')
parser.add_argument('--labels-path-name', default='labels', type=str,
                    help='Specify name labels folder in the derivative repository.',)
parser.add_argument('--session-name', default='', type=str,
                    help='name of the session to add before anat.',)

args = parser.parse_args()

root = Path(args.path_data)
train_ratio, test_ratio = args.split
path_out = Path(os.path.join(os.path.abspath(args.path_out), f'Dataset{args.dataset_number}_{args.dataset_name}'))

# create individual directories for train and test images and labels
path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

train_images, train_labels, test_images, test_labels = [], [], [], []

if args.session_name != '':
    anat_path = args.session_name + '/anat'
else:
    anat_path = 'anat'


if __name__ == '__main__':

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    # set the random number generator seed
    rng = np.random.default_rng(args.seed)

    # # Get all subjects from participants.tsv
    # subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
    # subjects_tsv = subjects_df['participant_id'].values.tolist()
    # logger.info(f"Total number of subjects in the tsv file: {len(subjects_tsv)}")

    # get the list of subjects in the root directory 
    subjects = [subject for subject in os.listdir(root) if subject.startswith('sub-')]
    logger.info(f"Total number of subjects in the root directory: {len(subjects)}")
    # sort the subjects list
    subjects.sort()

    # Get the training and test splits
    if test_ratio == 0:
        train_subjects = subjects
        test_subjects = None
    else:
        train_subjects, test_subjects = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)
    # rng.shuffle(train_subjects)

    # # list of contrasts: T2w, T1w, T2star, MTon, MToff, DWI
    # contrasts = ['dwi', 'flip-1_mt-on_MTS', 'flip-2_mt-off_MTS', 'T1w', 'T2star', 'T2w']

    train_ctr, test_ctr = 0, 0
    for subject in subjects:

        if subject in train_subjects:

            internal_ctr = 0
            # loop over all contrasts
            for contrast in args.contrasts:
                if contrast == 'task-tactile_bold' or contrast == 'task-rest_bold' or contrast == 'task-motor_bold' or contrast == 'task-thermal_bold' or contrast == 'task-rest_bold' or contrast == 'task-BilatMotorThermal_bold':
                    anat_path = 'func'
                if args.session_name != '':
                    contrast = args.session_name + '_' + contrast
                if contrast != 'dwi':
                    subject_images_path = os.path.join(root, subject, anat_path)
                    subject_labels_path = os.path.join(root, 'derivatives', args.labels_path_name, subject, anat_path)

                    subject_image_file = os.path.join(subject_images_path, f"{subject}_{contrast}.nii.gz")
                    subject_label_file = os.path.join(subject_labels_path, f"{subject}_{contrast}_{args.label_suffix}.nii.gz")

                    # check if both image and label files exist
                    if not (os.path.exists(subject_image_file) and os.path.exists(subject_label_file)):
                        print(f"Skipping train subject {subject}'s contrast {contrast} as either image or label does not exist.")
                        continue

                    # create the new convention names for nnunet
                    sub_name = str(Path(subject_image_file).name).split('_')[0] # + '_' + str(Path(subject_image_file).name).split('_')[1]
                    subject_image_file_nnunet = os.path.join(path_out_imagesTr,f"{args.dataset_name}_{sub_name}-{contrast}_{train_ctr:03d}_0000.nii.gz")
                    subject_label_file_nnunet = os.path.join(path_out_labelsTr,f"{args.dataset_name}_{sub_name}-{contrast}_{train_ctr:03d}.nii.gz")

                    # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                    os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                    os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

                    # binarize the label file
                    binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

                    # increment the counters
                    train_ctr += 1
                    internal_ctr += 1

                else:
                    subject_images_path = os.path.join(root, subject, 'dwi')
                    subject_labels_path = os.path.join(root, 'derivatives', args.labels_path_name, subject, 'dwi')

                    subject_image_file = os.path.join(subject_images_path, f"{subject}_rec-average_{contrast}.nii.gz")
                    subject_label_file = os.path.join(subject_labels_path, f"{subject}_rec-average_{contrast}_{args.label_suffix}.nii.gz")

                    # check if both image and label files exist
                    if not (os.path.exists(subject_image_file) and os.path.exists(subject_label_file)):
                        print(f"Skipping train subject {subject}'s contrast {contrast} as either image or label does not exist.")
                        continue

                    # create the new convention names for nnunet
                    sub_name = str(Path(subject_image_file).name).split('_')[0] # + '_' + str(Path(subject_image_file).name).split('_')[1]
                    subject_image_file_nnunet = os.path.join(path_out_imagesTr,f"{args.dataset_name}_{sub_name}-{contrast}_{train_ctr:03d}_0000.nii.gz")
                    subject_label_file_nnunet = os.path.join(path_out_labelsTr,f"{args.dataset_name}_{sub_name}-{contrast}_{train_ctr:03d}.nii.gz")

                    # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                    os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                    os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

                    # binarize the label file
                    binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

                    # increment the counters
                    train_ctr += 1
                    internal_ctr += 1

            # print(f"Found {internal_ctr} contrasts for Subject {subject}.")

        elif subject in test_subjects:

            internal_ctr = 0
            # loop over all contrasts
            for contrast in args.contrasts:
                if args.session_name != '':
                    contrast = args.session_name + '_' + contrast
                if contrast != 'dwi':
                    subject_images_path = os.path.join(root, subject, anat_path)
                    subject_labels_path = os.path.join(root, 'derivatives', args.labels_path_name, subject, anat_path)

                    subject_image_file = os.path.join(subject_images_path, f"{subject}_{contrast}.nii.gz")
                    subject_label_file = os.path.join(subject_labels_path, f"{subject}_{contrast}_{args.label_suffix}.nii.gz")

                    # check if both image and label files exist
                    if not (os.path.exists(subject_image_file) and os.path.exists(subject_label_file)):
                        print(f"Skipping test subject {subject}'s contrast {contrast} as either image or label does not exist.")
                        continue

                    # create the new convention names for nnunet
                    sub_name = str(Path(subject_image_file).name).split('_')[0] # + '_' + str(Path(subject_image_file).name).split('_')[1]
                    subject_image_file_nnunet = os.path.join(path_out_imagesTs,f"{args.dataset_name}_{sub_name}-{contrast}_{test_ctr:03d}_0000.nii.gz")
                    subject_label_file_nnunet = os.path.join(path_out_labelsTs,f"{args.dataset_name}_{sub_name}-{contrast}_{test_ctr:03d}.nii.gz")

                    # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                    os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                    os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

                    # binarize the label file
                    binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

                    # increment the counters
                    test_ctr += 1
                    internal_ctr += 1

                else:
                    subject_images_path = os.path.join(root, subject, 'dwi')
                    subject_labels_path = os.path.join(root, 'derivatives', args.labels_path_name, subject, 'dwi')

                    subject_image_file = os.path.join(subject_images_path, f"{subject}_rec-average_{contrast}.nii.gz")
                    subject_label_file = os.path.join(subject_labels_path, f"{subject}_rec-average_{contrast}_{args.label_suffix}.nii.gz")

                    # check if both image and label files exist
                    if not (os.path.exists(subject_image_file) and os.path.exists(subject_label_file)):
                        print(f"Skipping test subject {subject}'s contrast {contrast} as either image or label does not exist.")
                        continue

                    # create the new convention names for nnunet
                    sub_name = str(Path(subject_image_file).name).split('_')[0] # + '_' + str(Path(subject_image_file).name).split('_')[1]
                    subject_image_file_nnunet = os.path.join(path_out_imagesTs,f"{args.dataset_name}_{sub_name}-{contrast}_{test_ctr:03d}_0000.nii.gz")
                    subject_label_file_nnunet = os.path.join(path_out_labelsTs,f"{args.dataset_name}_{sub_name}-{contrast}_{test_ctr:03d}.nii.gz")

                    # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                    os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                    os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

                    # binarize the label file
                    binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

                    # increment the counters
                    test_ctr += 1
                    internal_ctr += 1

        else:
            print("Skipping file, could not be located in the Train or Test splits split.", subject)

    logger.info(f"Number of training and validation subjects (including all contrasts): {train_ctr}")
    logger.info(f"Number of test subjects (including all contrasts): {test_ctr}")
    # assert train_ctr == len(train_subjects), 'No. of train/val images do not match'
    # assert test_ctr == len(test_subjects), 'No. of test images do not match'

    # c.f. dataset json generation
    # In nnUNet V2, dataset.json file has become much shorter. The description of the fields and changes
    # can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#datasetjson
    # this file can be automatically generated using the following code here:
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/dataset_conversion/generate_dataset_json.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()
    json_dict['name'] = args.dataset_name
    json_dict['description'] = args.dataset_name
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"
    json_dict['numTraining'] = train_ctr
    json_dict['numTest'] = test_ctr

    # The following keys are the most important ones. 
    """
    channel_names:
        Channel names must map the index to the name of the channel. For BIDS, this refers to the contrast suffix.
        {
            0: 'T1',
            1: 'CT'
        }
    Note that the channel names may influence the normalization scheme!! Learn more in the documentation.
    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }
        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!
    """

    json_dict['channel_names'] = {
        0: "acq-sag_T2w",
    }

    json_dict['labels'] = {
        "background": 0,
        "sc-seg": 1,
    }

    # Needed for finding the files correctly. IMPORTANT! File endings must match between images and segmentations!
    json_dict['file_ending'] = ".nii.gz"

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)