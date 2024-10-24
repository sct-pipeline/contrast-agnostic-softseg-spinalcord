"""
Loop across JSON sidecar files and nii headers in the input path and parse from them the following information:
    MagneticFieldStrength
    Manufacturer
    ManufacturerModelName
    ProtocolName
    PixDim
    SliceThickness

If JSON sidecar is not available, fetch only PixDim and SliceThickness from nii header.

The fetched information is saved to a CSV file (parsed_data.csv).

Example usage:
    python scripts/fetch_sequence_parameters.py -i /path/to/dataset -contrast T2w

Original Author: Jan Valosek
    
Adapted from: 
https://github.com/ivadomed/model_seg_sci/blob/1079d156c322f555cab38c846240ab936ba98afb/utils/fetch_sequence_parameters.py

Adapted by: Naga Karthik
"""

import os
import re
import glob
import json
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import nibabel as nib

LIST_OF_PARAMETERS = [
    'MagneticFieldStrength',
    'Manufacturer',
    'ManufacturerModelName',
    'ProtocolName',
    'RepetitionTime',
    'EchoTime',
    'InversionTime',
    ]


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Loop across JSON sidecar files in the input path and parse from them relevant information.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '--path-datalists', required=True, type=str,
        help='Path to datalist json containing filename of subjects included train/val/test splits. ' 
             '(i.e. the output of create_msd_data.py script). Output is stored in args.path_datalist/sequence_parameters'
    )

    return parser


def parse_json_file(file_path):
    """
    Read the JSON file and parse from it relevant information.
    :param file_path:
    :return:
    """

    file_path = file_path.replace('.nii.gz', '.json')

    # Read the JSON file, return dict with n/a if the file is empty
    try:
        with open(file_path) as f:
            data = json.load(f)
    except:
        print(f'WARNING: {file_path} is empty.')
        return {param: "n/a" for param in LIST_OF_PARAMETERS}

    # Initialize an empty dictionary to store the parsed information
    parsed_info = {}

    if 'sci-zurich' in file_path:
        # For sci-zurich, JSON file contains a list of dictionaries, each dictionary contains a list of dictionaries
        data = data['acqpar'][0]
    elif 'sci-colorado' in file_path:
        data = data

    # Loop across the parameters
    for param in LIST_OF_PARAMETERS:
        try:
            parsed_info[param] = data[param]
        except:
            parsed_info[param] = "n/a"

    return parsed_info


def parse_nii_file(file_path):
    """
    Read nii file header using nibabel and to get PixDim and SliceThickness.
    We are doing this because 'PixelSpacing' and 'SliceThickness' can be missing from the JSON file.
    :param file_path:
    :return:
    """

    _, contrast_id = fetch_participant_details(file_path)

    # Read the nii file, return dict with n/a if the file is empty
    try:
        img = nib.load(file_path)
        header = img.header
    except:
        print(f'WARNING: {file_path} is empty. Did you run git-annex get .?')
        return {param: "n/a" for param in ['PixDim', 'SliceThickness']}

    # Initialize an empty dictionary to store the parsed information
    parsed_info = {
        'Contrast': contrast_id,
        'PixDim': list(header['pixdim'][1:3]),
        'SliceThickness': float(header['pixdim'][3])
    }

    return parsed_info


def fetch_participant_details(input_string):
    """
    Fetch the participant_id from the input string
    :param input_string: input string or path, e.g. 'sub-5416_T2w_seg_nnunet'
    :return participant_id: subject id, e.g. 'sub-5416'
    """
    participant = re.search('sub-(.*?)[_/]', input_string)  # [_/] slash or underscore
    participant_id = participant.group(0)[:-1] if participant else ""  # [:-1] removes the last underscore or slash

    if 'data-multi-subject' in input_string:
        # NOTE: the preprocessed spine-generic dataset have a weird BIDS naming convention (due to how they were preprocessed)
        contrast_pattern =  r'.*_(space-other_T1w|space-other_T2w|space-other_T2star|flip-1_mt-on_space-other_MTS|flip-2_mt-off_space-other_MTS|rec-average_dwi).*'
    else:
        # TODO: add more contrasts as needed
        # contrast_pattern =  r'.*_(T1w|T2w|T2star|PSIR|STIR|UNIT1|acq-MTon_MTR|acq-dwiMean_dwi|acq-b0Mean_dwi|acq-T1w_MTR).*'
        contrast_pattern =  r'.*_(T1w|T2w|T2star|PSIR|STIR|UNIT1|T1map|inv-1_part-mag_MP2RAGE|inv-2_part-mag_MP2RAGE|acq-MTon_MTR|acq-dwiMean_dwi|acq-T1w_MTR).*'
    contrast = re.search(contrast_pattern, input_string)
    contrast_id = contrast.group(1) if contrast else ""


    return participant_id, contrast_id


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    datalists = [os.path.join(args.path_datalists, file) for file in os.listdir(args.path_datalists) if file.endswith('_seed50.json')]
    out_path = os.path.join(args.path_datalists, 'sequence_parameters')
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    for datalist in datalists:
        dataset_name = datalist.split('/')[-1].split('_')[1]

        if not os.path.exists(datalist):
            print(f'ERROR: {datalist} does not exist. Run create_msd_data.py script first.')

        # load json file
        with open(datalist, 'r') as f:
            data = json.load(f)
        
        list_of_files = []
        for split in ['train', 'validation', 'test']:
            for idx in range(len(data[split])):
                list_of_files.append(data[split][idx]["image"])

        # Initialize an empty list to store the parsed data
        parsed_data = []


        # Loop across JSON sidecar files in the input path
        for file in tqdm(list_of_files):
            # print(f'Parsing {file}')
            parsed_json = parse_json_file(file)
            parsed_header = parse_nii_file(file)
            # Note: **metrics is used to unpack the key-value pairs from the metrics dictionary
            parsed_data.append({'filename': file, **parsed_json, **parsed_header})

        # Create a pandas DataFrame from the parsed data
        df = pd.DataFrame(parsed_data)

        df['filename'] = df['filename'].apply(lambda x: x.replace('/home/GRAMES.POLYMTL.CA/u114716/datasets/', ''))

        # Save the DataFrame to a CSV file
        df.to_csv(os.path.join(out_path, f'{dataset_name}_parsed_data.csv'), index=False)
        print(f"Parsed data saved to {os.path.join(out_path, f'{dataset_name}_parsed_data.csv')}")

    # # For sci-paris, we do not have JSON sidecars --> we can fetch only PixDim and SliceThickness from nii header
    # if 'sci-paris' in dir_path:
    #     # Print the min and max values of the PixDim, and SliceThickness
    #     print(df[['PixDim', 'SliceThickness']].agg([np.min, np.max]))
    # else:
    #     # Remove rows with n/a values for MagneticFieldStrength
    #     df = df[df['MagneticFieldStrength'] != 'n/a']

    #     # Convert MagneticFieldStrength to float
    #     df['MagneticFieldStrength'] = df['MagneticFieldStrength'].astype(float)

    #     # Print the min and max values of the MagneticFieldStrength, PixDim, and SliceThickness
    #     print(df[['MagneticFieldStrength', 'PixDim', 'SliceThickness']].agg([np.min, np.max]))

    #     # Print unique values of the Manufacturer and ManufacturerModelName
    #     print(df[['Manufacturer', 'ManufacturerModelName']].drop_duplicates())
    #     # Print number of filenames for unique values of the Manufacturer
    #     print(df.groupby('Manufacturer')['filename'].nunique())
    #     # Print number of filenames for unique values of the MagneticFieldStrength
    #     print(df.groupby('MagneticFieldStrength')['filename'].nunique())


if __name__ == '__main__':
    main()