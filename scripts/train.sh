#!/bin/bash
# 
# This script does the following:
# 1. Creates a virtual environment `venv_monai` and installs the required dependencies
# 2. Generates a MSD-style datalist containing image/label pairs for training
# 3. Trains the contrast-agnostic soft segmentation model
# 4. Evaluates the model on the test set
# 
# Usage:
# bash train.sh <path_to_preprocessed_data> <contrast> <label_type> <path_to_train_yaml>
# 
# Examples:
# 1. Train a model on 'T1w' contrast with 'hard' labels
#       bash train.sh /path/to/spine-generic-processed/ t1w hard train.yaml
# 2. Train a model on 'all' contrasts with 'soft' labels
#       bash train.sh /path/to/spine-generic-processed/ all soft train.yaml
#
#


# Uncomment for full verbose
# set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Set the following variables to the desired values
# Path to the pre-processed spine-generic dataset
PATH_DATA=$1
CONTRAST=$2     # options: ["t1w", "t2w", "t2star", "mton", "mtoff", "dwi", "all"]
LABEL_TYPE=$3   # options: ["hard", "soft", "soft_bin"]
PATH_TRAIN_YAML=$4  # path to the yaml file containing the training configuration

PATH_DATALIST_OUT="../datalists/spine-generic/temp/seed15/"
# create folder if doesn't exist
if [ ! -d $PATH_DATALIST_OUT ]; then
  mkdir -p $PATH_DATALIST_OUT
fi

MODEL="nnunet"

# # Create virtual environment
# conda create -n venv_monai python=3.9 -y

# NOTE: running conda activate <env_name> errors out requesting conda init to be run, 
# the eval expression here makes it work without conda init.
# source: https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
eval "$(conda shell.bash hook)"
conda activate venv_monai

# # Install dependencies quietly
# pip install -r monai/requirements.txt --quiet

# Run the script to generate the datalist JSON file
python monai/create_msd_data.py \
  --path-data $PATH_DATA \
  --path-out $PATH_DATALIST_OUT \
  --path-joblib "monai/split_datasets_all_seed=15.joblib" \
  --contrast $CONTRAST \
  --label-type $LABEL_TYPE \
  --seed 15

echo "----------------------------------------"
echo "Training contrast-agnostic '$MODEL' model on '$CONTRAST' contrasts with '$LABEL_TYPE' labels"
echo "----------------------------------------"

# Train the model
python monai/main.py --model $MODEL --config $PATH_TRAIN_YAML --debug

# Deactivate the virtual environment
conda deactivate
