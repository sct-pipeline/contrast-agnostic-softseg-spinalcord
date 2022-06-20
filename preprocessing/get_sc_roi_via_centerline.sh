#!/bin/bash
#
# Preprocess data.
#
# Dependencies:
# - FSL (includes bet2): 5.0.11
# - SCT: 5.3.0
#
# Usage:
#   ./get_sc_roi_via_centerline.sh <SUBJECT>
#
#
# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/

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

# get starting time:
start=`date +%s`


# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy BIDS-required files to processed data folder (e.g. list of participants)
if [[ ! -f "participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv .
fi
if [[ ! -f "participants.json" ]]; then
  rsync -avzh $PATH_DATA/participants.json .
fi
if [[ ! -f "dataset_description.json" ]]; then
  rsync -avzh $PATH_DATA/dataset_description.json .
fi
if [[ ! -f "README" ]]; then
  rsync -avzh $PATH_DATA/README .
fi

# Copy source images
rsync -avzh $PATH_DATA/$SUBJECT .

# Copy segmentation GTs
mkdir -p derivatives/labels derivatives/labels_softseg
rsync -avzh $PATH_DATA/derivatives/labels/$SUBJECT derivatives/labels/.
rsync -avzh $PATH_DATA/derivatives/labels_softseg/$SUBJECT derivatives/labels_softseg/.

# (1) Go to subject folder for source images
cd ${SUBJECT}

# Define paths for all contrasts
file_t1w_onlyfile="${SUBJECT}_T1w"
file_t1w="anat/${SUBJECT}_T1w"
file_t2w_onlyfile="${SUBJECT}_T2w"
file_t2w="anat/${SUBJECT}_T2w"
file_t2star_onlyfile="${SUBJECT}_T2star"
file_t2star="anat/${SUBJECT}_T2star"

# Get spinal cord centerline for all contrasts
sct_get_centerline -i ${file_t1w}.nii.gz -c "t1" -method "optic" -o ${file_t1w_onlyfile}_centerline
sct_get_centerline -i ${file_t2w}.nii.gz -c "t2" -method "optic" -o ${file_t2w_onlyfile}_centerline
sct_get_centerline -i ${file_t2star}.nii.gz -c "t2s" -method "optic" -o ${file_t2star_onlyfile}_centerline

# Get spinal cord ROI for all contrasts
sct_create_mask -i ${file_t1w}.nii.gz -p centerline,${file_t1w_onlyfile}_centerline.nii.gz -o ${file_t1w_onlyfile}_roi.nii.gz
sct_create_mask -i ${file_t2w}.nii.gz -p centerline,${file_t2w_onlyfile}_centerline.nii.gz -o ${file_t2w_onlyfile}_roi.nii.gz
sct_create_mask -i ${file_t2star}.nii.gz -p centerline,${file_t2star_onlyfile}_centerline.nii.gz -o ${file_t2star_onlyfile}_roi.nii.gz
# TODO: Maybe some dilation?

# Compute the bounding box coordinates of spinal cord ROI mask for cropping for all contrasts
# NOTE: `fslstats -w returns the smallest ROI <xmin> <xsize> <ymin> <ysize> <zmin> <zsize> <tmin> <tsize> containing nonzero voxels
t1w_roi_bbox_coords=$(fslstats ${file_t1w_onlyfile}_roi.nii.gz -w)
t2w_roi_bbox_coords=$(fslstats ${file_t2w_onlyfile}_roi.nii.gz -w)
t2star_roi_bbox_coords=$(fslstats ${file_t2star_onlyfile}_roi.nii.gz -w)

# Apply spinal cord mask to all contrasts
fslmaths ${file_t1w}.nii.gz -mas ${file_t1w_onlyfile}_roi.nii.gz ${file_t1w_onlyfile}.nii.gz
fslmaths ${file_t2w}.nii.gz -mas ${file_t2w_onlyfile}_roi.nii.gz ${file_t2w_onlyfile}.nii.gz
fslmaths ${file_t2star}.nii.gz -mas ${file_t2star_onlyfile}_roi.nii.gz ${file_t2star_onlyfile}.nii.gz

# Crop the ROI based on spinal cord mask to minimize the input image size
fslroi ${file_t1w_onlyfile}.nii.gz ${file_t1w_onlyfile}.nii.gz $t1w_roi_bbox_coords
fslroi ${file_t2w_onlyfile}.nii.gz ${file_t2w_onlyfile}.nii.gz $t2w_roi_bbox_coords
fslroi ${file_t2star_onlyfile}.nii.gz ${file_t2star_onlyfile}.nii.gz $t2star_roi_bbox_coords

# The following files are the final images for the contrasts, which will be inputted to the model
# t1w -> ${file_t1w_onlyfile}.nii.gz
# t2w -> ${file_t2w_onlyfile}.nii.gz
# t2star -> ${file_t2star_onlyfile}.nii.gz

# (2) Go to subject folder for original segmentation GTs
cd $PATH_DATA_PROCESSED/derivatives/labels/$SUBJECT

file_t1w_gt_onlyfile="${SUBJECT}_T1w_seg-manual"
file_t1w_gt="anat/${SUBJECT}_T1w_seg-manual"
file_t2w_gt_onlyfile="${SUBJECT}_T2w_seg-manual"
file_t2w_gt="anat/${SUBJECT}_T2w_seg-manual"
file_t2star_gt_onlyfile="${SUBJECT}_T2star_seg-manual"
file_t2star_gt="anat/${SUBJECT}_T2star_seg-manual"

# Apply the spinal cord mask to all GTs for all contrasts
fslmaths ${file_t1w_gt}.nii.gz -mas $PATH_DATA_PROCESSED/$SUBJECT/${file_t1w_onlyfile}_roi ${file_t1w_gt_onlyfile}.nii.gz
fslmaths ${file_t2w_gt}.nii.gz -mas $PATH_DATA_PROCESSED/$SUBJECT/${file_t2w_onlyfile}_roi ${file_t2w_gt_onlyfile}.nii.gz
fslmaths ${file_t2star_gt}.nii.gz -mas $PATH_DATA_PROCESSED/$SUBJECT/${file_t2star_onlyfile}_roi ${file_t2star_gt_onlyfile}.nii.gz

# Crop the ROI based on spinal cord mask to minimize the GT image size
fslroi ${file_t1w_gt_onlyfile}.nii.gz ${file_t1w_gt_onlyfile}.nii.gz $t1w_roi_bbox_coords
fslroi ${file_t2w_gt_onlyfile}.nii.gz ${file_t2w_gt_onlyfile}.nii.gz $t2w_roi_bbox_coords
fslroi ${file_t2star_gt_onlyfile}.nii.gz ${file_t2star_gt_onlyfile}.nii.gz $t2star_roi_bbox_coords

# The following files are the final GTs for all contrasts, which will be inputted to the model
# t1w -> ${file_t1w_gt_onlyfile}.nii.gz
# t2w -> ${file_t2w_gt_onlyfile}.nii.gz
# t2star -> ${file_t2star_gt_onlyfile}.nii.gz

# (3) Go to subject folder for soft, super-duper segmentation GTs
cd $PATH_DATA_PROCESSED/derivatives/labels_softseg/$SUBJECT

file_t1w_soft_gt_onlyfile="${SUBJECT}_T1w_softseg"
file_t1w_soft_gt="anat/${SUBJECT}_T1w_softseg"
file_t2w_soft_gt_onlyfile="${SUBJECT}_T2w_softseg"
file_t2w_soft_gt="anat/${SUBJECT}_T2w_softseg"
file_t2star_soft_gt_onlyfile="${SUBJECT}_T2star_softseg"
file_t2star_soft_gt="anat/${SUBJECT}_T2star_softseg"

# Apply the spinal cord mask to all GTs for all contrasts
fslmaths ${file_t1w_soft_gt}.nii.gz -mas $PATH_DATA_PROCESSED/$SUBJECT/${file_t1w_onlyfile}_roi ${file_t1w_soft_gt_onlyfile}.nii.gz
fslmaths ${file_t2w_soft_gt}.nii.gz -mas $PATH_DATA_PROCESSED/$SUBJECT/${file_t2w_onlyfile}_roi ${file_t2w_soft_gt_onlyfile}.nii.gz
fslmaths ${file_t2star_soft_gt}.nii.gz -mas $PATH_DATA_PROCESSED/$SUBJECT/${file_t2star_onlyfile}_roi ${file_t2star_soft_gt_onlyfile}.nii.gz

# Crop the ROI based on spinal cord mask to minimize the GT image size
fslroi ${file_t1w_soft_gt_onlyfile}.nii.gz ${file_t1w_soft_gt_onlyfile}.nii.gz $t1w_roi_bbox_coords
fslroi ${file_t2w_soft_gt_onlyfile}.nii.gz ${file_t2w_soft_gt_onlyfile}.nii.gz $t2w_roi_bbox_coords
fslroi ${file_t2star_soft_gt_onlyfile}.nii.gz ${file_t2star_soft_gt_onlyfile}.nii.gz $t2star_roi_bbox_coords

# The following files are the final soft GTs for all contrasts, which will be inputted to the model
# t1w -> ${file_t1w_soft_gt_onlyfile}.nii.gz
# t2w -> ${file_t2w_soft_gt_onlyfile}.nii.gz
# t2star -> ${file_t2star_soft_gt_onlyfile}.nii.gz

# Go back to parent folder (i.e. get ready for next subject call!)
cd $PATH_DATA_PROCESSED

# TODO: Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------

# Go back to the root output path
cd $PATH_OUTPUT

# Create and populate clean data processed folder for training
PATH_DATA_PROCESSED_CLEAN="${PATH_DATA_PROCESSED}_clean"

# Copy over required BIDs files
mkdir -p $PATH_DATA_PROCESSED_CLEAN $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/

rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/participants.* $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/README $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/derivatives/

# Refer to lines 105-108 to know which preprocessed files to copy
# MR images for T1w
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/${file_t1w_onlyfile}.nii.gz $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file_t1w_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/${file_t1w_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file_t1w_onlyfile}.json
# MR images for T2w
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/${file_t2w_onlyfile}.nii.gz $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file_t2w_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/${file_t2w_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file_t2w_onlyfile}.json
# MR images for T2star
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/${file_t2star_onlyfile}.nii.gz $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file_t2star_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/${file_t2star_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file_t2star_onlyfile}.json

# Refer to lines 130-133 to know which preprocessed GTs to copy
mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives $PATH_DATA_PROCESSED_CLEAN/derivatives/labels $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg
mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT} $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg/${SUBJECT}
mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg/${SUBJECT}/anat

# Preprocessed GTs for T1w
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/${file_t1w_gt_onlyfile}.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/${file_t1w_gt_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/anat/${file_t1w_gt_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/${file_t1w_gt_onlyfile}.json
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels_softseg/${SUBJECT}/${file_t1w_soft_gt_onlyfile}.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg/${SUBJECT}/anat/${file_t1w_soft_gt_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels_softseg/${SUBJECT}/anat/${file_t1w_soft_gt_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg/${SUBJECT}/anat/${file_t1w_soft_gt_onlyfile}.json
# Preprocessed GTs for T2w
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/${file_t2w_gt_onlyfile}.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/${file_t2w_gt_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/anat/${file_t2w_gt_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/${file_t2w_gt_onlyfile}.json
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels_softseg/${SUBJECT}/${file_t2w_soft_gt_onlyfile}.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg/${SUBJECT}/anat/${file_t2w_soft_gt_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels_softseg/${SUBJECT}/anat/${file_t2w_soft_gt_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg/${SUBJECT}/anat/${file_t2w_soft_gt_onlyfile}.json
# Preprocessed GTs for T2star
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/${file_t2star_gt_onlyfile}.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/${file_t2star_gt_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/anat/${file_t2star_gt_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/${file_t2star_gt_onlyfile}.json
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels_softseg/${SUBJECT}/${file_t2star_soft_gt_onlyfile}.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg/${SUBJECT}/anat/${file_t2star_soft_gt_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels_softseg/${SUBJECT}/anat/${file_t2star_soft_gt_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg/${SUBJECT}/anat/${file_t2star_soft_gt_onlyfile}.json


# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"