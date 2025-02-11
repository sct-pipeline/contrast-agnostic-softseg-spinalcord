"""
This script takes as input a dataset, a list of difficult subjects, and contrasts to create a folder with them.
It does the following:
1. createa a folder "difficult-cases" in output path, copies the subjects to this folder
2. outputs a yaml file with the list of subjects segregated based on the datasets
NOTE: this script keeps the BIDS folder structure intact so that it could be used with sct_run_batch

The idea is to build a dataset of difficult subjects for benchmarking our segmentation models (current ones and new ones we develop)

Author: Naga Karthik

"""

import os
import yaml
import argparse
import glob
import subprocess

def get_parser():

    parser = argparse.ArgumentParser(description='Create a list of difficult subjects')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='root path to the dataset containing the subjects')
    parser.add_argument('--include', type=str, required=True, nargs='+',
                        help='list of difficult subjects to be included')
    parser.add_argument('--contrasts', type=str, required=True, nargs='+',
                        help='list of contrasts to be copied for each subject. If "all" is provided, all files will be copied')
    parser.add_argument('--path-out', type=str, required=True,
                        help='path to the output directory where the folder "difficult-cases" will be created')
    return parser

def main():

    args = get_parser().parse_args()

    path_dataset = args.dataset
    contrasts = args.contrasts
    path_out = os.path.join(args.path_out, 'difficult-cases-temp')
    if not os.path.exists(path_out):
        print(f'Creating folder at: {path_out}')
        os.makedirs(path_out, exist_ok=True)
    else:
        print(f'Folder already exists! Adding subjects to: {path_out}')

    # check if a yaml file already exists and load it into the dictionary
    if os.path.exists(os.path.join(path_out, 'difficult_cases.yaml')):
        with open(os.path.join(path_out, 'difficult_cases.yaml'), 'r') as file:
            difficult_cases_dict = yaml.load(file, Loader=yaml.FullLoader)
    else:
        difficult_cases_dict = {}

    # loop through all subjects
    for subject_id in args.include:

        dataset = os.path.basename(path_dataset)
        if dataset not in difficult_cases_dict:
            difficult_cases_dict[dataset] = []

        subject_path = os.path.join(path_dataset, subject_id)

        for contrast in contrasts:
            
            if contrast == 'all':
                # find all image files for all contrasts
                files = subprocess.run(f'find {subject_path} -name "*.nii.gz"', shell=True, capture_output=True, text=True).stdout.split('\n')
            else:
                # find all image files for the contrast
                files = subprocess.run(f'find {subject_path} -name "*{contrast}*.nii.gz"', shell=True, capture_output=True, text=True).stdout.split('\n')

            if len(files) == 1 and not files[0]:
                print(f'No files found for {subject_id} contrast {contrast}')
            else:
                # get the relative path between subject_id and file
                for file in files:
                    if file:
                        relative_path = os.path.relpath(file, subject_path)
                        rel_path_with_subject = os.path.join(subject_id, relative_path)
                        if rel_path_with_subject in difficult_cases_dict[dataset]:
                            print(f'Subject {subject_id} contrast {contrast} already exists in the difficult cases list')
                            continue
                        # copy the file to the output directory
                        output_path = os.path.join(path_out, rel_path_with_subject)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        print(f'Copying {file} to {output_path}')
                        subprocess.run(f'cp {file} {output_path}', shell=True)
                        difficult_cases_dict[dataset].append(rel_path_with_subject)
        
    # save the difficult cases list
    with open(os.path.join(path_out, 'difficult_cases.yaml'), 'w') as file:
        documents = yaml.dump(difficult_cases_dict, file)

if __name__ == '__main__':
    main()


