#!/bin/bash
# Post-training script for evaluating morphometric drift of a lifelong learning contrast-agnostic spinal cord segmentation model. 
# It assumes that git-annex and Spinal Cord Toolbox (https://spinalcordtoolbox.com/) are installed. 
# 
# This standalone script that performs the following tasks:
# 1. Clones the spine-generic `data-multi-subject` (https://github.com/spine-generic/data-multi-subject) dataset
# 2. Downloads the data only for the subjects listed in `test_split_for_csa_drift_monitoring.yaml`
# 3. Runs a batch analysis (sct_run_batch) to compute the spinal cord cross-sectional area (CSA) on the test 
#   subjects downloaded in step 2 using a given version of the contrast-agnostic model.
# 4. Stores the output CSV files in the current directory, to be uploaded as part of the next release of the model
# 
# Usage:
#   bash compute_morphometrics_spine_generic.sh

# Exit immediately if a command exits with a non-zero status
set -e

# ==============================
# DEFINE GLOBAL VARIABLES
# ==============================
# # Path to the repository; all scripts will be relative to this path
# PATH_REPO="/home/GRAMES.POLYMTL.CA/${USER}/contrast-agnostic/contrast-agnostic-softseg-spinalcord"

# get current working directory before doing anything else
CWD=${PWD}

# Path to the output folder; the data, model, results, etc. will be stored in this folder
# PATH_OUTPUT="/home/GRAMES.POLYMTL.CA/${USER}/contrast-agnostic/test-post-training-script"
PATH_OUTPUT="csa-analysis"

# Path to the folder where the model exists, will be copied to the output folder PATH_OUTPUT
# for testing purposes, replace the PATH_MODEL with the path to the model downloaded from the latest release
MODEL_URL="https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases/download/v3.1/model_contrast_agnostic_20250123.zip"

# NOTE: To be compatible previous releases of the models, and to be able to automatically generate the
# morphometric plots after a new model is released, the model folder after copying will have the following
# syntax: ${PATH_OUTPUT}/model_${VERSION_TO_BE_RELEASED}
# Example: If the latest release in the contrast-agnostic repo points to the tag v3.1,
# then the next version to be released is v3.2. 
VERSION_TO_BE_RELEASED=v3.1

# Number of parallel processes to run (choose a smaller number as inference is run only on 1 gpu)
NUM_WORKERS=4

# # ID of the GPU to run inference on {0,1,2,3}
# CUDA_DEVICE_ID=3

# # Create folders 
# mkdir -p "${PATH_OUTPUT}"
# mkdir -p "${PATH_OUTPUT}/model_${VERSION_TO_BE_RELEASED}"

# # Copy the model to the output folder
# cp -r ${PATH_MODEL}/* ${PATH_OUTPUT}/model_${VERSION_TO_BE_RELEASED}

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
    "sub-barcelona06" "sub-beijingPrisma01" "sub-beijingPrisma02" "sub-brnoCeitec04" "sub-brnoUhb01")   # test only 5 for debugging
#     "sub-cardiff03" "sub-cmrra02" "sub-cmrra05" "sub-cmrrb01" "sub-cmrrb03"
#     "sub-cmrrb05" "sub-fslAchieva04" "sub-fslPrisma01" "sub-fslPrisma02" "sub-fslPrisma04"
#     "sub-fslPrisma05" "sub-geneva03" "sub-juntendo750w01" "sub-juntendo750w02" "sub-juntendo750w03"
#     "sub-juntendo750w06" "sub-milan03" "sub-mniS03" "sub-mountSinai01" "sub-nottwil01"
#     "sub-nottwil04" "sub-nwu01" "sub-oxfordFmrib06" "sub-oxfordFmrib09" "sub-oxfordFmrib10"
#     "sub-oxfordOhba01" "sub-oxfordOhba05" "sub-pavia02" "sub-pavia05" "sub-queensland01"
#     "sub-sherbrooke02" "sub-sherbrooke05" "sub-sherbrooke06" "sub-stanford04" "sub-strasbourg04"
#     "sub-tehranS03" "sub-tokyoIngenia05" "sub-ubc06" "sub-ucl02" "sub-unf04"
#     "sub-vuiisAchieva04" "sub-vuiisIngenia03" "sub-vuiisIngenia04" "sub-vuiisIngenia05"
# )

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

echo "=============================="
echo "Downloading model from URL ${MODEL_URL} ..."
echo "=============================="

# # Download the model zip file
# wget -O ${PATH_OUTPUT}/model_${VERSION_TO_BE_RELEASED}.zip ${MODEL_URL}
# # Unzip the model
# unzip ${PATH_OUTPUT}/model_${VERSION_TO_BE_RELEASED}.zip -d ${PATH_OUTPUT}/model_${VERSION_TO_BE_RELEASED}
# # Remove the zip file after extraction
# rm ${PATH_OUTPUT}/model_${VERSION_TO_BE_RELEASED}.zip

sct_deepseg -install seg_sc_contrast_agnostic -custom-url ${MODEL_URL}

echo "Model download complete."

# ==============================
# RUN BATCH ANALYSIS
# NOTE: this section piggybacks on the sct_run_batch argument provided by SCT
# Instead of providing a config file for batch processing script, we will provide the input arguments below
# ==============================

echo "=============================="
echo "Running batch analysis ..."
echo "=============================="

# Run batch processing
todays_date=$(date +"%Y%m%d")
path_out_run_batch=${PATH_OUTPUT}/${todays_date}__results_csa__model_${VERSION_TO_BE_RELEASED}
echo ${path_out_run_batch}

sct_run_batch -path-data ${PATH_OUTPUT}/data-multi-subject \
    -path-output ${path_out_run_batch} \
    -jobs ${NUM_WORKERS} \
    -script scripts/compute_csa.sh \
    -include-list ${TEST_SUBJECTS[@]}
    # if running directly from sct_deepseg, we don't need any script_args
    # -script-args "${CUDA_DEVICE_ID} ${PATH_REPO}/nnUnet/run_inference_single_subject.py ${PATH_OUTPUT}/model_${VERSION_TO_BE_RELEASED}" \
    # -include-list ${TEST_SUBJECTS[@]}


# echo "=============================="
# echo "Moving results to ${PATH_REPO}/csa_qc_evaluation_spine_generic ..."
# echo "=============================="

# cp -r ${path_out_run_batch}/results/csa_c2c3.csv ${PATH_REPO}/csa_qc_evaluation_spine_generic

# echo "=============================="
# echo "Morphometrics computation done!"
# echo "Upload the CSV file along with the release to compare CSA drift with respect to previous models."
# echo "=============================="
