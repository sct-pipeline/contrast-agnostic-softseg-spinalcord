#!/bin/bash
# 
# Usage:
#   bash compute_morphometrics_spine_generic.sh

# Exit immediately if a command exits with a non-zero status
set -e

# ==============================
# DEFINE GLOBAL VARIABLES
# ==============================

# get current working directory before doing anything else
CWD=${PWD}

# Path to the output folder; the data, model, results, etc. will be stored in this folder
# PATH_OUTPUT="/home/GRAMES.POLYMTL.CA/${USER}/contrast-agnostic/test-post-training-script"
PATH_OUTPUT="csa-analysis"

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# ==============================
# DOWNLOAD DATA
# ==============================

echo "=============================="
echo "Downloading test data ..."
echo "=============================="

# Clone the dataset and initialize a git annex repository
url_dataset="https://github.com/spine-generic/data-multi-subject"
tag="r20250310"
clone_folder="${PATH_OUTPUT}/data-multi-subject"

# Ref: https://stackoverflow.com/questions/36498981/shell-dont-fail-git-clone-if-folder-already-exists/36499031#36499031
if [ ! -d "$clone_folder" ] ; then
    echo "Cloning dataset ..."
    git clone --branch "$tag" "$url_dataset" "$clone_folder"
    # change directory
    cd $clone_folder
else
    echo "Dataset already exists, skipping cloning."
fi

# Initialize an empty git-annex repository 
git annex init

# # Extract test subjects and store them in an array
# readarray -t TEST_SUBJECTS < <(python -c 'import yaml, sys; 
# test_subjects = yaml.safe_load(open(sys.argv[1]))["test"]; 
# for subject in test_subjects: print(subject)' "${PATH_REPO}/scripts/spine_generic_test_split_for_csa_drift_monitoring.yaml")

TEST_SUBJECTS=(
    "sub-barcelona06" "sub-beijingPrisma01" "sub-beijingPrisma02" "sub-brnoCeitec04" "sub-brnoUhb01"   # test only 5 for debugging
    "sub-cardiff03" "sub-cmrra02" "sub-cmrra05" "sub-cmrrb01" "sub-cmrrb03"
    "sub-cmrrb05" "sub-fslAchieva04" "sub-fslPrisma01" "sub-fslPrisma02" "sub-fslPrisma04"
    "sub-fslPrisma05" "sub-geneva03" "sub-juntendo750w01" "sub-juntendo750w02" "sub-juntendo750w03"
    "sub-juntendo750w06" "sub-milan03" "sub-mniS03" "sub-mountSinai01" "sub-nottwil01"
    "sub-nottwil04" "sub-nwu01" "sub-oxfordFmrib06" "sub-oxfordFmrib09" "sub-oxfordFmrib10"
    "sub-oxfordOhba01" "sub-oxfordOhba05" "sub-pavia02" "sub-pavia05" "sub-queensland01"
    "sub-sherbrooke02" "sub-sherbrooke05" "sub-sherbrooke06" "sub-stanford04" "sub-strasbourg04"
    "sub-tehranS03" "sub-tokyoIngenia05" "sub-ubc06" "sub-ucl02" "sub-unf04"
    "sub-vuiisAchieva04" "sub-vuiisIngenia03" "sub-vuiisIngenia04" "sub-vuiisIngenia05"
)

# Download test split using git-annex
for subject in "${TEST_SUBJECTS[@]}"; do
    echo "Downloading: $subject"
    # download images
    git annex get "${subject}"
    # change current working directory to derivatives
    cd $PWD/derivatives
    # cd derivatives
    # download all kinds of labels
    git annex get $(find . -name "${subject}")
    # change back to root directory
    cd ..
done

echo "Dataset download complete."

# Return to the root directory of the repo
cd ${CWD}    # this will go to the root of the repository
