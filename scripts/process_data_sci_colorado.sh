#!/bin/bash
#
# Process data from sci-colorado
#     Crop all images.
# Authors: Sandrine Bédard

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# Uncomment for full verbose
set -x

# Immediately exit if error
#set -e -o pipefail --> will not enter in the loop if so...

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
segment_if_exists(){
  local file="$1"
  local contrast="$2"
  local contrast_for_seg="$3"
  # Find contrast
  if [[ $contrast == "./dwi/" ]]; then
    folder_contrast="dwi"
  else
    folder_contrast="anat"
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
    sct_deepseg_sc -i ${file}.nii.gz -c $contrast_for_seg -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${FILESEG}.nii.gz
    
  fi
}

# Go to anat folder where all structural data are located
cd ${SUBJECT}/anat/

# T1w
# ------------------------------------------------------------------------------

file_t1="${SUBJECT}_T1w"
segment_if_exists ${file_t1} 'anat' 't1'
file_t1_seg="${file_t1}_seg"

# T2w
# ------------------------------------------------------------------------------
file_t2="${SUBJECT}_T2w"
segment_if_exists ${file_t2} 'anat' 't2'
file_t2_seg="${file_t2}_seg"

file_path=$file_t2
fileseg="$file_t2_seg-manual"
# Dilate spinal cord segmentation
sct_maths -i ${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${fileseg}.nii.gz -dilate 7 -shape ball -o ${fileseg}_dilate.nii.gz
# Crop image 
sct_crop_image -i ${file_path}.nii.gz -m ${fileseg}_dilate.nii.gz -o ${file_path}_crop.nii.gz
# Crop segmentation
sct_crop_image -i ${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${fileseg}.nii.gz -m ${fileseg}_dilate.nii.gz -o ${fileseg}_crop.nii.gz

# Go back to root output folder
cd $PATH_OUTPUT
# Create and populate clean data processed folder for training
PATH_DATA_PROCESSED_CLEAN="${PATH_DATA_PROCESSED}_clean"
# Copy over required BIDs files
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/participants.* $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/README.md $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/
mkdir -p $PATH_DATA_PROCESSED_CLEAN $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat
# Put cropped image and json in cleaned dataset
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/${file_path}_crop.nii.gz $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file_path}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/${file_path}.json $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file_path}.json
# Move segmentation
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/${fileseg}_crop.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/${fileseg}.nii.gz
# Move json files of derivatives
rsync -avzh "${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${fileseg}.json" $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/anat/${fileseg}.json

# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
FILES_TO_CHECK=(
  "anat/${SUBJECT}_T1w.nii.gz"
  "anat/${SUBJECT}_T2w.nii.gz"
  "anat/${SUBJECT}_T2w_seg.nii.gz"
  "dwi/${SUBJECT}_T1w_seg.nii.gz"
)
pwd
for file in ${FILES_TO_CHECK[@]}; do
  if [[ ! -e $file ]]; then
    echo "${SUBJECT}/anat/${file} does not exist" >> $PATH_LOG/_error_check_output_files.log
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
