#!/bin/bash
#
# Generate QCs for T2w images:
#     - sagittal lesion QC
#     - single slice sagittal spinal cord QC (to check FOV coverage (C/Th/L))
#
# Dependencies (versions):
# - SCT 6.0 and higher
#
# Usage:
# sct_run_batch -script generate_qc.sh -path-data <PATH-TO-DATASET> -path-output <PATH-TO-OUTPUT> -jobs <num-cpu-cores>

# Lesion and SC labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/anat/

# With the following naming convention:
# file_gt="${file}_lesion-manual"
# file_seg="${file}_seg-manual"

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT


# Retrieve input params and other params
SUBJECT=$1
QC_DATASET=$2           # dataset name to generate QC for

echo "SUBJECT: ${SUBJECT}"
echo "QC_DATASET: ${QC_DATASET}"

# get starting time:
start=`date +%s`


# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

if [[ $QC_DATASET == "lumbar-epfl" ]]; then
    contrasts=("T2w")
    label_suffix="seg-manual"

elif [[ $QC_DATASET == "lumbar-vanderbilt" ]]; then
    contrasts=("acq-axial_T2star")
    label_suffix="label-SC_seg"

elif [[ $QC_DATASET == "nih-ms-mp2rage" ]]; then
    contrasts=("UNIT1")
    label_suffix="label-SC_seg"

elif [[ $QC_DATASET == "sct-testing-large" ]]; then
    contrasts="T1w T2star T2w acq-MTon_MTR"
    label_suffix="seg-manual"

elif [[ $QC_DATASET == "canproco" ]]; then
    contrasts="PSIR STIR T2w"
    label_suffix="seg-manual"

elif [[ $QC_DATASET == "spider-challenge-2023" ]]; then
    contrasts="acq-lowresSag_T1w acq-lowresSag_T2w acq-highresSag_T2w"
    label_suffix="label-SC_seg"

fi

PATH_DERIVATIVES="${PATH_DATA}/derivatives/labels/./${SUBJECT}/anat"
PATH_IMAGES="${PATH_DATA}/./${SUBJECT}/anat"

# Loop across contrasts
for contrast in ${contrasts}; do
    
    # NOTE: this replacement is cool because it automatically takes care of 'ses-XX' for longitudinal data
    file="${SUBJECT//[\/]/_}_${contrast}"

    # check if label exists in the dataset
    if [[ ! -f ${PATH_DERIVATIVES}/${file}_${label_suffix}.nii.gz ]]; then
        echo "Label File ${file}_${label_suffix}.nii.gz does not exist. Skipping..."
        
    else
        echo "Label File ${file}_${label_suffix}.nii.gz exists. Proceeding..."
        
        # copy labels
        rsync -Ravzh ${PATH_DERIVATIVES}/${file}_${label_suffix}.nii.gz .

        # copy source images
        rsync -Ravzh ${PATH_IMAGES}/${file}.nii.gz .

        # Generate QC report the GT spinal cord segmentation
        if [[ $QC_DATASET == "canproco" ]]; then
            # do sagittal qc
            sct_qc -i ${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file}.nii.gz -s ${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file}_${label_suffix}.nii.gz -d ${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file}_${label_suffix}.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject ${SUBJECT}
        else 
            sct_qc -i ${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file}.nii.gz -s ${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file}_${label_suffix}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
        fi
    fi

done

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"