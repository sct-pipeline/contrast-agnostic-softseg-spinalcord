#!/bin/bash
#
# Compute CSA on output segmentations. Input is spine-generic/data-multi-subject
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
set -e -o pipefail

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

# Check if manual label already exists. If it does, copy it locally. If it does
# not, perform labeling.
# NOTE: manual disc labels should go from C1-C2 to C7-T1.
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  # Update global variable with segmentation file name
  FILELABEL="${file}_labels-disc"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${FILELABEL}-manual.nii.gz"
  # Binarize softsegmentation to create labeled softseg
  #sct_maths -i ${file_seg}.nii.gz -bin 0.5 -o ${file_seg}_bin.nii.gz
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c t2
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c t2
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
file_t1w="${SUBJECT}_acq-T1w_MTS"
file_mton="${SUBJECT}_acq-MTon_MTS"
file_dwi_mean="${SUBJECT}_rec-average_dwi"
contrasts=($file_t1 $file_t2s $file_t1w $file_mton $file_dwi_mean)
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
echo "Contrasts are" ${inc_contrasts[@]}


# Copy predicted segmentation
rsync -avzh ${PATH_PRED_SEG}/${SUBJECT}_T2w_pred.nii.gz ./anat/
pred_seg_t2=${SUBJECT}_T2w_pred

# Put pred mask in the original image space
#sct_register_multimodal -i $pred_seg_t2 -d ./anat/$file_t2 -identity 1
#pred_seg_t2=${pred_seg_t2}_reg
# Create labeled segmentation of vertebral levels (only if it does not exist) 
label_if_does_not_exist ./anat/$file_t2 ./anat/$pred_seg_t2

file_t2_seg_labeled="${pred_seg_t2}_labeled"
file_t2_disc="${pred_seg_t2}_labeled_discs"
# Generate QC report to assess vertebral labeling
sct_qc -i ./anat/${file_t2}.nii.gz -s ${file_t2_seg_labeled}.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Compute average cord CSA between C2 and C3
sct_process_segmentation -i ./anat/${pred_seg_t2}.nii.gz -vert 2:3 -vertfile ${file_t2_seg_labeled}.nii.gz -o ${PATH_RESULTS}/csa-SC_c2c3_${SUBJECT}.csv -append 1

for file_path in "${inc_contrasts[@]}";do

# Find contrast to do compute CSA

  if [[ $file_path == *"T1w"* ]];then
    contrast_seg="t1"
  elif [[ $file_path == *"T2star"* ]];then
    contrast_seg="t2s"
  elif [[ $file_path == *"T1w_MTS"* ]];then
    contrast_seg="t1"
  elif [[ $file_path == *"MTon_MTS"* ]];then
    contrast_seg="t2s"
  elif [[ $file_path == *"dwi"* ]];then
    contrast_seg="dwi"
  fi

  type=$(find_contrast $file_path)
  file=${file_path/#"$type"}
  fileseg=${file_path}_seg
  # Check if file exists (pred file)
  if [[ -f ${PATH_PRED_SEG}${file}_pred.nii.gz ]];then
    rsync -avzh ${PATH_PRED_SEG}${file}_pred.nii.gz ${type}
    pred_seg=${SUBJECT}_${contrast_seg}_pred
  fi

  # Register contrast to T2w to get warping field 
  # Registration
  # ------------------------------------------------------------------------------
  sct_register_multimodal -i ${file_path}.nii.gz -d ./anat/${file_t2}.nii.gz -iseg ${fileseg}_pad.nii.gz -dseg ./anat/${file_t2_seg}.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares,iter=10,poly=2 -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${file_path}_reg.nii.gz
  warping_field=${type}warp_${file_t2}2${file}
  sct_apply_transfo -i ${file_t2_disc}.nii.gz -d ./anat/${pred_seg}.nii.gz -w ${warping_field}.nii.gz -x linear -o ${fileseg}_reg.nii.gz


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
