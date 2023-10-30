#!/bin/bash
#
# Run QC report on prediction masks of T2w axial images of dcm-zurich
# Usage:
#   ./run_qc_prediction_dcm_zurich_ax.sh <SUBJECT>
#
#
# Authors: Sandrine Bédard
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

set -x
# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1
PATH_PRED_SEG=$2
path_source=$(dirname $PWD)
PATH_SCRIPT=$path_source


# get starting time:
start=`date +%s`

# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd ${PATH_DATA_PROCESSED}
# Copy list of participants in processed data folder
if [[ ! -f "participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv .
fi
# Copy list of participants in resutls folder
if [[ ! -f $PATH_RESULTS/"participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv $PATH_RESULTS/"participants.tsv"
fi

# Copy source images
# Copy session if specified
if [[ $SUBJECT == *"ses"* ]];then
    rsync -Ravzh $PATH_DATA/./$SUBJECT .
else
    rsync -avzh $PATH_DATA/$SUBJECT .
fi


# SCRIPT STARTS HERE
# ==============================================================================
# Go to anat folder where all structural data are located
cd ${SUBJECT}/anat
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_sub="${SUBJECT//[\/]/_}"
file="${file_sub}_T2w"

# Run segmentation
python $PATH_SCRIPT/monai/run_inference_single_image.py --path-img ${file}.nii.gz --path-out . --chkp-path ~/duke/temp/muena/contrast-agnostic/final_monai_model/nnunet_nf=32_DS=1_opt=adam_lr=0.001_AdapW_CCrop_bs=2_64x192x320_20230918-2253
sct_maths -i ${file}_pred.nii.gz -bin 0.5 -o ${file}_pred_bin.nii.gz
sct_qc -i ${file}.nii.gz -s ${file}_pred_bin.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
sct_deepseg_sc -i ${file}.nii.gz -c t2 -qc ${PATH_QC} -qc-subject ${SUBJECT}


file="${file_sub}_T1w"

# Run segmentation
python $PATH_SCRIPT/monai/run_inference_single_image.py --path-img ${file}.nii.gz --path-out . --chkp-path ~/duke/temp/muena/contrast-agnostic/final_monai_model/nnunet_nf=32_DS=1_opt=adam_lr=0.001_AdapW_CCrop_bs=2_64x192x320_20230918-2253
sct_maths -i ${file}_pred.nii.gz -bin 0.5 -o ${file}_pred_bin.nii.gz
sct_qc -i ${file}.nii.gz -s ${file}_pred_bin.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
sct_deepseg_sc -i ${file}.nii.gz -c t1 -qc ${PATH_QC} -qc-subject ${SUBJECT}



# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
