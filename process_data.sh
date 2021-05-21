#!/bin/bash
#
# Process data. Note. Input data are already resample and reoriented to RPI.
#
# Usage:
#   ./process_data.sh <SUBJECT>
#
# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/anat/
#
# Authors: Sandrine Bédard

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
rsync -avzh $PATH_DATA/$SUBJECT .
# Go to anat folder where all structural data are located
cd ${SUBJECT}/anat/

# FUNCTIONS
# ==============================================================================

segment_if_does_not_exist(){
  local file="$1"
  local contrast="$2"
  folder_contrast="anat"

  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${folder_contrast}/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg_sc -i ${file}.nii.gz -c $contrast -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# T1w
# ------------------------------------------------------------------------------
file_t1="${SUBJECT}_T1w"

# Reorient to RPI and resample to 0.8mm iso (supposed to be the effective resolution)
sct_image -i ${file_t1}.nii.gz -setorient RPI -o ${file_t1}_RPI.nii.gz
sct_resample -i ${file_t1}_RPI.nii.gz -mm 1x1x1 -o ${file_t1}_RPI_r.nii.gz # Do we want the resolution of T2?
file_t1="${file_t1}_RPI_r"

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t1 "t1"
file_t1_seg=$FILESEG

# T2w
# ------------------------------------------------------------------------------
file_t2="${SUBJECT}_T2w"

sct_image -i ${file_t2}.nii.gz -setorient RPI -o ${file_t2}_RPI.nii.gz
sct_resample -i ${file_t2}_RPI.nii.gz -mm 0.8x0.8x0.8 -o ${file_t2}_RPI_r.nii.gz 
file_t2="${file_t2}_RPI_r"

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t2 "t2"
file_t2_seg=$FILESEG

# T2s
# ------------------------------------------------------------------------------
#file_t2s="${SUBJECT}_T2star"
# Compute root-mean square across 4th dimension (if it exists), corresponding to all echoes in Philips scans.
#sct_maths -i ${file_t2s}.nii.gz -rms t -o ${file_t2s}_rms.nii.gz
#file_t2s="${file_t2s}_rms"
# 
#sct_image -i ${file_t2s}.nii.gz -setorient RPI -o ${file_t2s}_RPI.nii.gz
# sct_resample -i ${file_t2}_RPI.nii.gz -mm 0.8x0.8x0.8 -o ${file_t2}_RPI_r.nii.gz | NO resampeling??
# Segment spinal cord (only if it does not exist)
#segment_if_does_not_exist $file_t2s "t2s" 
#file_t2s_seg=$FILESEG

# Registration
# ------------------------------------------------------------------------------
file_t2_mask="${file_t2_seg}_dil"
ImageMath 3 ${file_t2_mask}.nii.gz MD ${file_t2_seg}.nii.gz 40

# Crop image for faster computing ??
#sct_crop_image -i ${file_t2}.nii.gz -m ${file_t2_mask}.nii.gz -o ${file_t2}_crop.nii.gz
#file_t2="${file_t2}_crop"

sct_register_multimodal -i ${file_t1}.nii.gz -d ${file_t2}.nii.gz -iseg ${file_t1_seg}.nii.gz -dseg ${file_t2_seg}.nii.gz -m ${file_t2_mask}.nii.gz -param step=1,type=im,algo=slicereg,slicewise=1,metric=CC,iter=10,shrink=4:step=2,type=im,algo=slicereg,slicewise=1,metric=CC,iter=10,shrink=2 -x spline -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Register T1w cord segmentation to T2w
sct_apply_transfo -i ${file_t1_seg}.nii.gz -d ${file_t2_seg}.nii.gz -w warp_${file_t1}2${file_t2}.nii.gz -x nn -o ${file_t1_seg}_reg.nii.gz
file_t1_seg="${file_t1_seg}_reg"

sct_qc -i ${file_t1}.nii.gz -s ${file_t1_seg}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}


# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
