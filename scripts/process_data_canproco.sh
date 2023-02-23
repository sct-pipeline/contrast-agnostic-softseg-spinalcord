#!/bin/bash
#
# Process data from canproco
#     Crop all images.
# Authors: Louis-François Bouchard, Sandrine Bédard

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail --> will not enter in the loop if so...

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1

# get starting time:
start=`date +%s`

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED
# Copy list of participants in processed data folder
if [[ ! -f "participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv .
fi
if [[ ! -f "participants.json" ]]; then
  rsync -avzh $PATH_DATA/participants.json .
fi
if [[ ! -f "dataset_description.json" ]]; then
  rsync -avzh $PATH_DATA/dataset_description.json .
fi
if [[ ! -f "README.md" ]]; then
  rsync -avzh $PATH_DATA/README.md .
fi
# Copy list of participants in results folder
if [[ ! -f $PATH_RESULTS/"participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv $PATH_RESULTS/"participants.tsv"
fi
# Copy source images
rsync -avzh $PATH_DATA/$SUBJECT .


# FUNCTIONS
# ==============================================================================
segment_if_does_not_exist(){
  local file="$1"
  local contrast="$2"
  local contrast_for_seg="$3"
  local propseg = $4

  # Find contrast
  if [[ $contrast == "./dwi/" ]]; then
    folder_contrast="dwi"
  else
    folder_contrast="ses-M0/anat"
  fi

  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${folder_contrast}/${FILESEG}-manual.nii.gz"


  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."

    rsync -avzh $FILESEGMANUAL "${FILESEG}.nii.gz"
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Manual segmentation not found."
    # Segment spinal cord
    if propseg; then
      sct_propseg -i ${file}.nii.gz -c $contrast_for_seg -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${FILESEG}.nii.gz
    else
      sct_deepseg_sc -i ${file}.nii.gz -c $contrast_for_seg -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${FILESEG}.nii.gz
    fi
  fi
}

# Go to anat folder where all structural data are located
cd ${SUBJECT}/ses-M0/anat/

file_session = "_ses-M0"

# T1w
# ------------------------------------------------------------------------------

file_t1="${SUBJECT}${file_session}_T1w_MTS"
segment_if_does_not_exist ${file_t1} 'anat' 't1' False
file_t1_seg="${file_t1}_seg"

# T2w
# ------------------------------------------------------------------------------
file_t2="${SUBJECT}${file_session}_T2w"
segment_if_does_not_exist ${file_t2} 'anat' 't2' False
file_t2_seg="${file_t2}_seg"

file_path=$file_t2
fileseg="$file_t2_seg-manual"

# MTon_MTS
# ------------------------------------------------------------------------------

# Segmentation settings from: https://github.com/ivadomed/canproco/issues/4#issuecomment-1430250475
file_mton="${SUBJECT}${file_session}_acq_MTon_MTS"
segment_if_does_not_exist ${file_mton} 'anat' 't2s' False
file_mton_seg="${file_mton}_seg"

# STIR
# ------------------------------------------------------------------------------

# Segmentation settings from: https://github.com/ivadomed/canproco/issues/4#issuecomment-1430250475
file_STIR="${SUBJECT}${file_session}_STIR"
segment_if_does_not_exist ${file_STIR} 'anat' 't2' False
file_STIR_seg="${file_STIR}_seg"

# PSIR
# ------------------------------------------------------------------------------

# Segmentation settings from: https://github.com/ivadomed/canproco/issues/4#issuecomment-1430250475
file_PSIR="${SUBJECT}${file_session}_PSIR"
segment_if_does_not_exist ${file_PSIR} 'anat' 't1' True
file_PSIR_seg="${file_PSIR}_seg"

# T2star
# ------------------------------------------------------------------------------

# Segmentation settings from: https://github.com/ivadomed/canproco/issues/4#issuecomment-1430250475
file_t2s="${SUBJECT}${file_session}_T2star"
segment_if_does_not_exist ${file_t2s} 'anat' 't2s' False
file_t2s_seg="${file_t2s}_seg"


# ------------------------------------------------------------------------------
# Crop image 
# Reorient to RPI and resample to 0.8mm isotropic voxel (supposed to be the effective resolution)
sct_image -i ${file_path}.nii.gz -setorient RPI -o ${file_path}_raw.nii.gz
sct_resample -i ${file_path}_raw.nii.gz -mm 0.8x0.8x0.8 -o ${file_path}.nii.gz # replace files with preprocessed
sct_crop_image -i ${file_path}.nii.gz -dilate 7 -m ${PATH_DATA}/derivatives/labels/${SUBJECT}/ses-M0/anat/${fileseg}.nii.gz -o ${file_path}_crop.nii.gz
# Crop segmentation
sct_image -i ${PATH_DATA}/derivatives/labels/${SUBJECT}/ses-M0/anat/${fileseg}.nii.gz -setorient RPI -o ${PATH_DATA}/derivatives/labels/${SUBJECT}/ses-M0/anat/${fileseg}.nii.gz
sct_resample -i ${PATH_DATA}/derivatives/labels/${SUBJECT}/ses-M0/anat/${fileseg}.nii.gz -mm 0.8x0.8x0.8 -o ${PATH_DATA}/derivatives/labels/${SUBJECT}/ses-M0/anat/${fileseg}.nii.gz # replace files with preprocessed
sct_crop_image -i ${PATH_DATA}/derivatives/labels/${SUBJECT}/ses-M0/anat/${fileseg}.nii.gz -dilate 7 -m ${PATH_DATA}/derivatives/labels/${SUBJECT}/ses-M0/anat/${fileseg}.nii.gz -o ${fileseg}_crop.nii.gz

# Go back to root output folder
cd $PATH_OUTPUT
# Create and populate clean data processed folder for training
PATH_DATA_PROCESSED_CLEAN="${PATH_DATA_PROCESSED}_clean"
# Copy over required BIDs files
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/participants.* $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/README.md $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/
mkdir -p $PATH_DATA_PROCESSED_CLEAN $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/ses-M0/anat $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-M0/anat
# Put cropped image and json in cleaned dataset
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/ses-M0/anat/${file_path}_crop.nii.gz $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/ses-M0/anat/${file_path}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/ses-M0/anat/${file_path}.json $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/ses-M0/anat/${file_path}.json
# Move segmentation
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/ses-M0/anat/${fileseg}_crop.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-M0/anat/${fileseg}.nii.gz
# Move json files of derivatives
rsync -avzh "${PATH_DATA}/derivatives/labels/${SUBJECT}/ses-M0/anat/${fileseg}.json" $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-M0/anat/${fileseg}.json

# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
FILES_TO_CHECK=(
  "ses-M0/anat/${SUBJECT}_T1w.nii.gz"
  "ses-M0/anat/${SUBJECT}_T2w.nii.gz"
  "ses-M0/anat/${SUBJECT}_T2w_seg.nii.gz"
)
pwd
for file in ${FILES_TO_CHECK[@]}; do
  if [[ ! -e $file ]]; then
    echo "${SUBJECT}/ses-M0/anat/${file} does not exist" >> $PATH_LOG/_error_check_output_files.log
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
