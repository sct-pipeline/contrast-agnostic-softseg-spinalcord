"""
This script:
  1. Clones different datasets from data.neuro.polymtl.ca. To download praxis datasets, 
  ensure that you have the access credentials and download the dataset manually from spineimage.ca
  2. Gets the git commit ID of the datasets and saves it to git_branch_commit.log

Usage:
    python 01_download_data.py -ofolder <PATH_TO_FOLDER_WHERE_DATA_WILL_BE_DOWNLOADED>

Authors: Jan Valosek (adapted by Naga Karthik)
"""

import os
import sys
import subprocess
import argparse
import yaml

# Add the parent directory of the script to the Python path
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from utils import get_git_branch_and_commit
# from utils.utils import SITES_DICT, get_git_branch_and_commit


def download_dataset(dataset_name, dataset_commit):
    if os.path.isdir(dataset_name):
        # Already cloned — fetch and checkout the requested commit
        os.chdir(dataset_name)
        subprocess.run(["git", "fetch", "--all"])
    else:
        # Fresh clone
        if dataset_name == 'data-multi-subject':
            subprocess.run(["git", "clone", f"https://github.com/spine-generic/{dataset_name}"])
        else:
            subprocess.run(["git", "clone", f"git@data.neuro.polymtl.ca:datasets/{dataset_name}"])
        os.chdir(dataset_name)
        subprocess.run(["git", "annex", "init"])
        subprocess.run(["git", "annex", "dead", "here"])

    # Checkout the specific commit
    subprocess.run(["git", "checkout", f"{dataset_commit}"])

    # Get the git commit ID of the dataset
    dataset_path = os.getcwd()
    branch, commit = get_git_branch_and_commit(dataset_path)
    with open(f"{PATH_DATA}/git_branch_commit.log", "a") as log_file:
        log_file.write(f"{dataset_name}: git-{branch}-{commit}\n")

    # # Download nii files from git-annex
    # files_t2 = subprocess.run(["find", ".", "-name", "*T2w*.nii.gz"], capture_output=True, text=True).stdout.splitlines()
    # files_t2 = [file for file in files_t2 if "STIR" not in file]
    # subprocess.run(["git", "annex", "get"] + files_t2)
    # os.chdir("..")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets from spineimage.ca")
    parser.add_argument("--ofolder",
                        metavar='DIR_NAME',
                        required=True,
                        type=str,
                        help="Path to the folder where data will be downloaded")
    parser.add_argument("--dataset",
                        required=True,
                        type=str,
                        help="Name of the dataset to be cloned")
    parser.add_argument('--path-datasplits', type=str, default=None,
                        help='Path to the datasplits folder containing predefined datasplits (used to fetch dataset commit)')
    args = parser.parse_args()

    PATH_DATA = os.path.abspath(os.path.expanduser(args.ofolder))
    os.chdir(PATH_DATA)

    # get commit of the dataset
    with open(os.path.join(args.path_datasplits, f"datasplit_{args.dataset}_seed50.yaml"), 'r') as file:
        datasplits = yaml.safe_load(file)
        dataset_commit = datasplits['dataset_version_commit']

    # for site, dataset_name in SITES_DICT.items():
    download_dataset(args.dataset, dataset_commit)