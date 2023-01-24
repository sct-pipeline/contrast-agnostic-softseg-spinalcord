#!/bin/bash
#
# Compute CSA on output segmentations. Input is data_processed_clean
#
# Usage:
#   ./compute_csa.sh <SUBJECT>
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
#set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1
PATH_PRED_SEG=$2

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

# FUNCTIONS
# ==============================================================================

# Check if manual label already exists. If it does, copy it locally.
# NOTE: manual disc labels should go from C1-C2 to C7-T1.
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  local ofolder="$3"
  # Update global variable with segmentation file name
  FILELABEL="${file}_discs"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${FILELABEL}.nii.gz"
  
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c t2 -ofolder $ofolder
  else
    echo "Not found. cannot compute CSA."
  fi
}

find_contrast(){
  local file="$1"
  local dwi="dwi"
  if echo "$file" | grep -q "$dwi"; then
    echo  "./${dwi}/"
  else
    echo "./anat/"
  fi
}


# SCRIPT STARTS HERE
# ==============================================================================
# Go to anat folder where all structural data are located
cd ${SUBJECT}

# Initialize filenames
file_t1="${SUBJECT}_T1w"
file_t2="${SUBJECT}_T2w"
file_t2s="${SUBJECT}_T2star"
file_t1w="${SUBJECT}_flip-2_mt-off_MTS"
file_mton="${SUBJECT}_flip-1_mt-on_MTS"
file_dwi_mean="${SUBJECT}_rec-average_dwi"
contrasts=($file_t1 $file_t2 $file_t2s $file_t1w $file_mton $file_dwi_mean)
inc_contrasts=()

# Check available contrasts
# ------------------------------------------------------------------------------

# Check if a list of images to exclude was passed.
if [ -z "$EXCLUDE_LIST" ]; then
  EXCLUDE=""
else
  EXCLUDE=$(yaml $PATH_SCRIPT/${EXCLUDE_LIST} "['FILES_REG']")
fi

for contrast in "${contrasts[@]}"; do
  type=$(find_contrast $contrast)
  if echo "$EXCLUDE" | grep -q "$contrast"; then
    echo "$contrast found in exclude list.";
  else
    if [[ -f "${type}${contrast}.nii.gz" ]]; then
      inc_contrasts+=(${type}${contrast})
    else
      echo "$contrast not found, excluding it."
    fi
  fi

done
echo "Contrasts are ${inc_contrasts[@]}"

for file_path in "${inc_contrasts[@]}";do
  # Find contrast to do compute CSA
  if [[ $file_path == *"flip-2_mt-off_MTS"* ]];then
    contrast_seg="flip-2_mt-off_MTS"
  elif [[ $file_path == *"T2star"* ]];then
    contrast_seg="T2star"
  elif [[ $file_path == *"T2w"* ]];then
    contrast_seg="T2w"
  elif [[ $file_path == *"T1w"* ]];then
    contrast_seg="T1w"
  elif [[ $file_path == *"flip-1_mt-on_MTS"* ]];then
    contrast_seg="flip-1_mt-on_MTS"
  elif [[ $file_path == *"dwi"* ]];then
    contrast_seg="dwi"
  fi

  type=$(find_contrast $file_path)
  file=${file_path/#"$type"}
  fileseg=${file_path}_seg
  # Check if file exists (pred file)
  if [[ -f ${PATH_PRED_SEG}${file}_pred.nii.gz ]];then
    rsync -avzh ${PATH_PRED_SEG}${file}_pred.nii.gz ${file_path}_pred.nii.gz
    pred_seg=${file_path}_pred

    # Remove 4th dimension
    sct_image -i ${pred_seg}.nii.gz -split t
    pred_seg=${pred_seg}_T0000

    # Create QC for pred mask
    sct_qc -i ${file_path}.nii.gz -s ${pred_seg}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

    # Get manual hard GT to get labeled segmentation
    FILESEG="${file_path}_seg"
    FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${FILESEG}-manual.nii.gz"
    echo
    echo "Looking for manual segmentation: $FILESEGMANUAL"
    if [[ -e $FILESEGMANUAL ]]; then
      echo "Found! Using manual segmentation."
      rsync -avzh $FILESEGMANUAL "${FILESEG}.nii.gz"
    fi
    # Create labeled segmentation of vertebral levels (only if it does not exist) 
    label_if_does_not_exist $file_path $FILESEG $type

    file_seg_labeled="${FILESEG}_labeled"
    # Generate QC report to assess vertebral labeling
    sct_qc -i ${file_path}.nii.gz -s ${file_seg_labeled}.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}

    # Compute average cord CSA between C2 and C3
    sct_process_segmentation -i ${pred_seg}.nii.gz -vert 2:3 -vertfile ${file_seg_labeled}.nii.gz -o ${PATH_RESULTS}/csa_pred_${contrast_seg}.csv -append 1
  else
    echo "Pred mask not found"
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
