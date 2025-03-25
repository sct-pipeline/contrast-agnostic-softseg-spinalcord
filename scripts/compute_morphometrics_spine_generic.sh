#!/bin/bash
# Post-training script for computing morphometrics on spine-generic dataset using lifelong learning contrast-agnostic 
# spinal cord segmentation model. 
# This script is one of the steps in the automated GitHub actions for computing spinal cord morphometrics (CSA)
# 
# This script performs the following tasks:
# 1. Downloads the model using sct_deepseg -install seg_sc_contrast_agnostic -custom-url <url>
# 2. Runs a batch analysis (sct_run_batch) to compute the spinal cord cross-sectional area (CSA) on a 
# mini-batch of test subjects obtained as input.
# 3. Moves the logs/ and results/ to the an output folder
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

# get list of subjects as input
TEST_SUBJECTS=($1)
echo "Running analysis on ${TEST_SUBJECTS[@]}"

# Path to the output folder; the data, model, results, etc. will be stored in this folder
PATH_OUTPUT="csa-analysis"

# Path to the folder where the model exists, will be copied to the output folder PATH_OUTPUT
# for testing purposes, replace the PATH_MODEL with the path to the model downloaded from the latest release
MODEL_URL=$2
echo "Using model at: ${MODEL_URL}"
# MODEL_URL="https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases/download/v3.1/model_contrast_agnostic_20250123.zip"

# Get model version
MODEL_VERSION=$(echo "$MODEL_URL" | sed -E 's#.*/download/([^/]+)/.*#\1#')

# Number of parallel processes to run (choose a smaller number as inference is run only on 1 gpu)
NUM_WORKERS=4

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

echo "=============================="
echo "Downloading model from URL ${MODEL_URL} ..."
echo "=============================="

sct_deepseg spinalcord -install -custom-url ${MODEL_URL}

echo "Model download complete."

# ==============================
# RUN BATCH ANALYSIS
# NOTE: this section piggybacks on the sct_run_batch argument provided by SCT
# ==============================

echo "=============================="
echo "Running batch analysis ..."
echo "=============================="

# Run batch processing
path_out_run_batch=${PATH_OUTPUT}/batch_processing_results
echo ${path_out_run_batch}

sct_run_batch -path-data data-multi-subject \
    -path-output ${path_out_run_batch} \
    -jobs ${NUM_WORKERS} \
    -script scripts/compute_csa.sh \
    -script-args "${MODEL_VERSION}" \
    -include-list ${TEST_SUBJECTS[@]}


echo "=============================="
echo "Copying log and results folders to ${CWD}/logs_results ..."
echo "=============================="

mkdir -p ${CWD}/logs_results
cp -r ${path_out_run_batch}/log ${CWD}/logs_results
cp -r ${path_out_run_batch}/results ${CWD}/logs_results
# NOTE: this copying is done so that it is easy to find these folders outside of the script to be uploaded by GH Actions

# Go back to the current working directory at the beginning
cd ${CWD}
