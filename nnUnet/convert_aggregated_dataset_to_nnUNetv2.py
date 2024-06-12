import os
import json
import argparse
from pathlib import Path
from loguru import logger
import shutil
from collections import OrderedDict
from utils import Image

def get_parser():

    parser = argparse.ArgumentParser(description='Code for Code for creating the aggregated contrast-agnostic dataset '
                                    'in nnUNetv2 format.')

    parser.add_argument('--path-datalists', required=True, type=str, help='Path to folder containing datalist json files.')
    parser.add_argument('--path-out', type=str, help='Path to the output directory where dataset json is saved')
    parser.add_argument('-dnum', '--dataset-num', required=True, type=int, help='Dataset number.')
    parser.add_argument('-dname', '--dataset-name', required=True, type=str, help='Dataset name.')

    return parser

def main():

    args = get_parser().parse_args()
    path_out = Path(os.path.join(os.path.abspath(args.path_out), f'Dataset{args.dataset_num}_{args.dataset_name}'))

    # create individual directories for train and test images and labels
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

    # make the directories
    Path(path_out).mkdir(parents=True, exist_ok=True)
    Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    datalists = [os.path.join(args.path_datalists, f) for f in os.listdir(args.path_datalists) if f.endswith('_seed50.json')]

    train_ctr, test_ctr = 0, 0
    for datalist in datalists[1:2]:
        logger.info(f'Processing datalist: {datalist}')
        with open(datalist) as f:
            data = json.load(f)

        for key in data.keys():
            if key in ['train', 'validation']:
                for item in data[key]:
                    image_path = Path(item['image'])
                    label_path = Path(item['label'])
                    image_name = image_path.name.replace('.nii.gz', '')
                    label_suffix = label_path.name.split(image_name)[1]
                    label_name = label_path.name.replace(label_suffix, '')

                    # ensure that both image and label files exist
                    if not (os.path.exists(image_path) and os.path.exists(label_path)):
                        logger.error(f"Image or label file does not exist: {image_path}, {label_path}")
                        continue

                    subject_image_file_nnunet = os.path.join(path_out_imagesTr, f"{args.dataset_name}_{image_name}_{train_ctr:03d}_0000.nii.gz")
                    subject_label_file_nnunet = os.path.join(path_out_labelsTr, f"{args.dataset_name}_{label_name}_{train_ctr:03d}.nii.gz")

                    # os.symlink(image_path, path_out_imagesTr / image_name)
                    # os.symlink(label_path, path_out_labelsTr / label_name)
                    shutil.copy(image_path, subject_image_file_nnunet)
                    shutil.copy(label_path, subject_label_file_nnunet)

                    # convert the image and label to RPI using the Image class
                    image = Image(subject_image_file_nnunet)
                    image.change_orientation("RPI")
                    image.save(subject_image_file_nnunet, verbose=False)

                    label = Image(subject_label_file_nnunet)
                    label.change_orientation("RPI")
                    label.save(subject_label_file_nnunet, verbose=False)

                    train_ctr += 1

            elif 'test' in key:
                for item in data[key]:
                    image_path = Path(item['image'])
                    label_path = Path(item['label'])
                    image_name = image_path.name.replace('.nii.gz', '')
                    label_suffix = label_path.name.split(image_name)[1]
                    label_name = label_path.name.replace(label_suffix, '')

                    # ensure that both image and label files exist
                    if not (os.path.exists(image_path) and os.path.exists(label_path)):
                        logger.error(f"Image or label file does not exist: {image_path}, {label_path}")
                        continue

                    # os.symlink(image_path, path_out_imagesTs / image_name)
                    # os.symlink(label_path, path_out_labelsTs / label_name)
                    subject_image_file_nnunet = os.path.join(path_out_imagesTs, f"{args.dataset_name}_{image_name}_{test_ctr:03d}_0000.nii.gz")
                    subject_label_file_nnunet = os.path.join(path_out_labelsTs, f"{args.dataset_name}_{label_name}_{test_ctr:03d}.nii.gz")

                    shutil.copy(image_path, subject_image_file_nnunet)
                    shutil.copy(label_path, subject_label_file_nnunet)

                    # convert the image and label to RPI using the Image class
                    image = Image(subject_image_file_nnunet)
                    image.change_orientation("RPI")
                    image.save(subject_image_file_nnunet, verbose=False)

                    label = Image(subject_label_file_nnunet)
                    label.change_orientation("RPI")
                    label.save(subject_label_file_nnunet, verbose=False)

                    test_ctr += 1

    logger.info(f"Number of training and validation subjects (including all contrasts): {train_ctr}")
    logger.info(f"Number of test subjects (including all contrasts): {test_ctr}")

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


if __name__ == '__main__':
    main()