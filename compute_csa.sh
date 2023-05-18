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
file_sub="${SUBJECT//[\/]/_}"

# Check if file exists (pred file)
for file_pred in ${PATH_PRED_SEG}/*; do
  if [[ $file_pred == *$file_sub* ]];then
      echo " File found, running QC report $file_pred"
      # Find if anat or dwi
      file_seg_basename=${file_pred##*/}
      echo $file_seg_basename
      type=$(find_contrast $file_pred)

      # rsync prediction mask
      rsync -avzh $file_pred ${type}/$file_seg_basename
      prefix="spineGNoCropSoftAvgBin_"
      file_image=${file_seg_basename#"$prefix"}
      echo $file_image
      file_image="${file_image::-11}"  # Remove X.nii.gz since teh number X varies
      # split with "-"
      arrIN=(${file_image//-/ })
      if [[ $type == *"dwi"* ]];then
        contrast="rec-average_dwi"  # original image name
        contrast_csv="dwi"
      else
        contrast=${file_image#${arrIN[0]}"-"}  # remove sub-
        contrast=${contrast#${arrIN[1]}"-"}  # remove sub-id
        contrast_csv=$contrast
      fi
      file_image=${arrIN[0]}"-"${arrIN[1]}"_"${contrast}
      echo $file_image

      pred_seg=${type}${file_seg_basename}

      # Create QC for pred mask
      sct_qc -i ${type}${file_image}.nii.gz -s ${pred_seg} -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

      # Get manual hard GT to get labeled segmentation
      FILESEG="${type}${file_image}_seg"
      FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${FILESEG}-manual.nii.gz"
      echo
      echo "Looking for manual segmentation: $FILESEGMANUAL"
      if [[ -e $FILESEGMANUAL ]]; then
        echo "Found! Using manual segmentation."
        rsync -avzh $FILESEGMANUAL "${FILESEG}.nii.gz"
      fi
      # Create labeled segmentation of vertebral levels (only if it does not exist) 
      label_if_does_not_exist ${type}${file_image} $FILESEG $type

      file_seg_labeled="${FILESEG}_labeled"
      # Generate QC report to assess vertebral labeling
      sct_qc -i ${type}${file_image}.nii.gz -s ${file_seg_labeled}.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}

      # Compute average cord CSA between C2 and C3
      sct_process_segmentation -i ${pred_seg} -vert 2:3 -vertfile ${file_seg_labeled}.nii.gz -o ${PATH_RESULTS}/csa_pred_${contrast_csv}.csv -append 1
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
