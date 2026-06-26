#!/bin/bash
# This script is used for training contrast-agnostic v3.0 and also provides the option to extend the
# contrast-agnostic spinal cord segmentation model with new datasets. It achieves the following:
# 1. Clones the datasets from NeuroPoly's git-annex server.
# 2. Creates datalists (i.e. json files with image/label pairs) based on pre-defined or random dataset splits
# 3. Converts the json files for each dataset into one aggregated dataset in the nnUNet format
#    (with sc_crop detection-based cropping applied to each image/label pair)
# 4. Runs nnUNet preprocessing and training based on the defined configurations (2D/3D).

set -e

# ====================================
# USER-DEFINED PATHS — edit these before running
# ====================================

# Full path to the contrast-agnostic repository
PATH_REPO="/home/quentinr/contrast-agnostic-softseg-spinalcord"

# Full path to the folder where datasets will be downloaded
PATH_DATA_BASE="/home/quentinr/datasets_contrast_agnostic_retraining"

# Full path to the folder where datalists (JSON files) will be saved
PATH_OUT_BASE="/home/quentinr/datalists"

# Full path to nnUNet directories (set nnUNet_raw / nnUNet_preprocessed / nnUNet_results)
PATH_NNUNET_BASE="/home/quentinr/nnunet-v2"

# Full path to the SCT bin directory (needed for sct_* commands)
PATH_SCT_BIN="/home/quentinr/spinalcordtoolbox/bin"


# ====================================
# VARIABLES FOR DATASET CREATION
# ====================================

# Set seed for reproducibility. Note seed=50 was used train the contrast-agnostic model
# If you're using a different seed, note that you cannot use the predefined random dataset splits (more info below)
SEED=50

# List of datasets to train on
# NOTE 1: the following datasets were used for training the contrast-agnostic v3.0 model
# https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases/tag/v3.0
# NOTE 2: training on praxis acute SCI data requires special access to `spineimage.ca`. Because this is different from
# the usual downloading from git-annex, this script does not support downloading praxis data. To train contrast-agnostic model
# download the dataset manually and store it in PATH_DATA_BASE (see below)

DATASETS=("data-multi-subject" "basel-mp2rage" "canproco" \
            "lumbar-epfl" "lumbar-vanderbilt" "dcm-brno" "dcm-zurich" "dcm-zurich-lesions" "dcm-zurich-lesions-20231115" \
            "sci-paris" "sci-zurich" "sci-colorado" "sct-testing-large" \
            "site_006" "site_007"
            )
# for debugging purposes, test the script on 1 dataset from the above list
# DATASETS=("data-multi-subject")

# Path to the output folder where the dataset in MSD-style format will be saved as json files with image/label pairs
# and other dataset-related statistics. To keep track of the experiments, date is also appended as a prefix or suffix
# Example: 20260524-sc-crop
folder_name=$(date +"%Y%m%d")-sc-crop
PATH_OUT_DATALISTS="${PATH_OUT_BASE}/${folder_name}"

# Path to yml file containing subjects to include. These subjects are curated to be of good quality after visual QC'ing
# Always include this file, when reproducing and also when adding new datasets
PATH_INCLUDE_SUBJECTS=${PATH_REPO}/subjects_to_include.yml


# ====================================
# VARIABLES FOR NNUNET TRAINING
# ====================================

# Path to store the converted dataset (ideally the ${nnUNet_raw} folder once nnUNet is installed)
PATH_NNUNET_RAW="${PATH_NNUNET_BASE}/nnUNet_raw"

# Path to the nnUNet results folder (ideally ${nnUNet_results})
PATH_NNUNET_RESULTS="${PATH_NNUNET_BASE}/nnUNet_results"

export nnUNet_raw=${PATH_NNUNET_RAW}
export nnUNet_preprocessed="${PATH_NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results=${PATH_NNUNET_RESULTS}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export PATH="${PATH_SCT_BIN}:${PATH}"

# Name and number/id of the dataset to be referenced by nnunet
DATASET_NAME="ContrastAgnosticScCrop"
DATASET_NUMBER=1000          # this refers to the `-d` argument when training nnunet models

# Name of the nnUNet trainer variant
# NOTE: contrast-agnostic v3.0 model used the default trainer defined below
NNUNET_TRAINER="nnUNetTrainer"
# NNUNET_TRAINER="nnUNetTrainer_5epochs"

# Name of the plans file. Recommended to keep the default one below unless you want to train
# other models given in nnunet's model suite
NNUNET_PLANS_FILE="nnUNetPlans"

# Type/Kernel of the model. for 2D training, use "2d"; for 3D training, use "3d_fullres"
# configurations=("2d" "3d_fullres")
configurations=("3d_fullres")

# Number of cross-validation folds to run the model on. nnUNet by default allows training on 5 folds
# folds=(0 1 2 3 4)
folds=(0)

# GPU ID to use for training the model
cuda_visible_devices=0

# Required on Blackwell (sm_120+) GPUs: disables torch.compile to avoid Triton compilation crashes
export TORCHDYNAMO_DISABLE=1


# ====================================
# STEP CONTROL — auto-detects where to resume, or override with START_STEP
# Steps: 1=clone  2=datalists  3=convert  4=preprocess  5=train
# Usage: bash train_contrast_agnostic.sh              (auto-resume)
#        START_STEP=4 bash train_contrast_agnostic.sh (force step 4)
#        START_STEP=4 END_STEP=4 bash train_contrast_agnostic.sh (step 4 only)
# ====================================
END_STEP=${END_STEP:-5}

if [ -z "${START_STEP}" ]; then
    NNUNET_RESULTS_DIR="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"
    NNUNET_PREPROCESSED_DIR="${nnUNet_preprocessed}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"
    NNUNET_RAW_DIR="${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"

    if [ -d "${NNUNET_RESULTS_DIR}/nnUNetTrainer__nnUNetPlans__${configurations[0]}" ]; then
        START_STEP=5
    elif [ -f "${NNUNET_PREPROCESSED_DIR}/nnUNetPlans.json" ]; then
        START_STEP=5
    elif [ -d "${NNUNET_RAW_DIR}" ]; then
        START_STEP=4
    elif [ -d "${PATH_OUT_DATALISTS}" ] && [ -n "$(ls ${PATH_OUT_DATALISTS}/*.json 2>/dev/null)" ]; then
        START_STEP=3
    elif [ -d "${PATH_DATA_BASE}/${DATASETS[0]}" ]; then
        START_STEP=2
    else
        START_STEP=1
    fi
    echo "-----------------------------------"
    echo "Auto-detected START_STEP=${START_STEP}"
    echo "-----------------------------------"
fi


# ====================================
# STEP 1 — CLONE DATASETS
# ====================================

if [ ${START_STEP} -le 1 ] && [ ${END_STEP} -ge 1 ]; then

    cd ${PATH_REPO}

    for dataset in ${DATASETS[@]}; do
        if [[ ${dataset} == site_* ]]; then
            echo "-----------------------------------"
            echo "Encountered possibly a PRAXIS dataset, checking if it is already downloaded ..."
            echo "-----------------------------------"
            if [[ ! -d "${PATH_DATA_BASE}/${dataset}" ]]; then
                echo "No dataset found in ${PATH_DATA_BASE}/${dataset}, please download the praxis dataset manually"
                exit
            else
                echo "Dataset found at ${PATH_DATA_BASE}/${dataset}, moving on to datalist creation ..."
            fi
        else
            echo "-----------------------------------"
            echo "Cloning ${dataset} dataset from git-annex ..."
            echo "-----------------------------------"
            python ${PATH_REPO}/nnUnet/01_clone_dataset.py \
                --ofolder ${PATH_DATA_BASE} \
                --dataset ${dataset} \
                --path-datasplits ${PATH_REPO}/datasplits
        fi
    done

    echo "-----------------------------------"
    echo "STEP 1 done — all datasets cloned."
    echo "-----------------------------------"

fi


# ====================================
# STEP 2 — CREATE DATALISTS
# ====================================

if [ ${START_STEP} -le 2 ] && [ ${END_STEP} -ge 2 ]; then

    cd ${PATH_REPO}

    for dataset in ${DATASETS[@]}; do
        echo "-----------------------------------"
        echo "Downloading data from git-annex and creating datalist for ${dataset} ..."
        echo "-----------------------------------"
        python ${PATH_REPO}/nnUnet/02_create_msd_data.py \
            --seed ${SEED} \
            --path-data ${PATH_DATA_BASE}/${dataset} \
            --path-out ${PATH_OUT_DATALISTS} \
            --include ${PATH_INCLUDE_SUBJECTS} \
            --use-predefined-splits \
            --path-datasplits ${PATH_REPO}/datasplits
    done

    echo "-----------------------------------"
    echo "STEP 2 done — datalists created and stored in ${PATH_OUT_DATALISTS}"
    echo "-----------------------------------"

fi


# ====================================
# STEP 3 — CONVERT TO NNUNET FORMAT (sc_crop detection-based cropping)
# ====================================

if [ ${START_STEP} -le 3 ] && [ ${END_STEP} -ge 3 ]; then

    echo "-----------------------------------"
    echo "Converting the datalists to nnUNetv2-specific format ..."
    echo "-----------------------------------"

    python ${PATH_REPO}/nnUnet/03_convert_msd_to_nnunet_reorient.py \
        --input ${PATH_OUT_DATALISTS} \
        --output ${PATH_NNUNET_RAW} \
        --taskname ${DATASET_NAME} \
        --tasknumber ${DATASET_NUMBER} \
        --workers 8

    echo "-----------------------------------"
    echo "STEP 3 done — converted datasets can be found in ${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}."
    echo "-----------------------------------"

fi


# ====================================
# STEP 4 — NNUNET PREPROCESSING
# ====================================

if [ ${START_STEP} -le 4 ] && [ ${END_STEP} -ge 4 ]; then

    for configuration in ${configurations[@]}; do
        echo "-----------------------------------"
        echo "Verifying dataset integrity and running preprocessing for ${configuration} configuration ..."
        echo "-----------------------------------"
        nnUNetv2_plan_and_preprocess -d ${DATASET_NUMBER} --verify_dataset_integrity -c ${configuration}
    done

    echo "-----------------------------------"
    echo "STEP 4 done — preprocessing completed."
    echo "-----------------------------------"

fi


# ====================================
# STEP 5 — NNUNET TRAINING
# ====================================

if [ ${START_STEP} -le 5 ] && [ ${END_STEP} -ge 5 ]; then

for configuration in ${configurations[@]}; do

    for fold in ${folds[@]}; do

        echo "-------------------------------------------"
        echo "Training on Fold $fold, Configuration $configuration ..."
        echo "-------------------------------------------"

        start=$(date +%s)

        CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ${DATASET_NUMBER} \
                                $configuration $fold -tr ${NNUNET_TRAINER} -p ${NNUNET_PLANS_FILE}

        echo ""
        echo "-------------------------------------------"
        echo "Training completed for Fold $fold, Configuration $configuration"
        echo "-------------------------------------------"

        end=`date +%s`
        runtime=$((end-start))
        echo
        echo "~~~"
        echo "Ran on:      `uname -nsr`"
        echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
        echo "~~~"
    done

done

echo "-------------------------------------------"
echo "STEP 5 done — training on all folds completed."
echo "Model can be found in ${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"
echo "-------------------------------------------"

fi
