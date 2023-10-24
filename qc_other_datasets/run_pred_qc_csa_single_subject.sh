#!/bin/bash
#
# Compute CSA on output segmentations. Input is data_processed_clean
# Adapted from compute_csa_nnunet.sh
#
# Usage:
#   ./compute_csa.sh <SUBJECT>
#
# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/anat/
#
# Authors: Naga Karthik 

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

# get starting time:
start=`date +%s`

# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short
# Save script path
# Get path derivatives
path_source=$(dirname $PWD)
PATH_SCRIPT=$path_source
SUBJECT=$1


echo "SUBJECT: ${SUBJECT}"

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Create directory
mkdir -p $PATH_DATA_PROCESSED/$SUBJECT

# Go to subject folder for source images
cd ${SUBJECT}

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




# Initialize filenames
file_t1="${SUBJECT}_T1w"
file_t2="${SUBJECT}_T2w"
file_t2s="${SUBJECT}_T2star"
file_t1w="${SUBJECT}_flip-2_mt-off_MTS"
file_mton="${SUBJECT}_flip-1_mt-on_MTS"
file_dwi_mean="${SUBJECT}_rec-average_dwi"
contrasts=($file_t1 $file_t2 $file_t2s $file_t1w $file_mton $file_dwi_mean)

for file_path in "${contrasts[@]}";do
  # Find contrast to do compute CSA
  if [[ $file_path == *"flip-2_mt-off_MTS"* ]];then
    contrast_seg="flip-2_mt-off_MTS"
    contrast="t1"
  elif [[ $file_path == *"T2star"* ]];then
    contrast_seg="T2star"
    contrast="t2s"
  elif [[ $file_path == *"T2w"* ]];then
    contrast_seg="T2w"
    contrast="t2"
  elif [[ $file_path == *"T1w"* ]];then
    contrast_seg="T1w"
    contrast="t1"  # For segmentation
  elif [[ $file_path == *"flip-1_mt-on_MTS"* ]];then
    contrast_seg="flip-1_mt-on_MTS"
    contrast="t2s"
  elif [[ $file_path == *"dwi"* ]];then
    contrast_seg="dwi"
    contrast="dwi"
  fi

  # Find if anat or dwi folder
  type=$(find_contrast $file_path)
  file=${file_path/#"$type"}  # add sub folder in file name
  file_path=${type}$file
  mkdir -p $PATH_DATA_PROCESSED/$SUBJECT/${type}
  # Copy source images
  # Note: we use '/./' in order to include the sub-folder 'ses-0X'
  rsync -Ravzh ${PATH_DATA}/${SUBJECT}/${file_path}.* .



  # Get manual hard GT to get labeled segmentation
  FILESEG="${file_path}_seg"
  FILESEGMANUAL="${PATH_DATA}/../data_processed/${SUBJECT}/${FILESEG}.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL "${FILESEG}.nii.gz"
  fi
  
  # Create labeled segmentation of vertebral levels (only if it does not exist) 
  label_if_does_not_exist $file_path $FILESEG $type
  file_seg_labeled="${FILESEG}_labeled"

  # Segment SC using different methods and compute ANIMA segmentation performance metrics
  python $PATH_SCRIPT/monai/run_inference_single_image.py --path-img ${file_path}.nii.gz --path-out . --chkp-path ~/duke/temp/muena/contrast-agnostic/final_monai_model/nnunet_nf=32_DS=1_opt=adam_lr=0.001_AdapW_CCrop_bs=2_64x192x320_20230918-2253
  sct_maths -i ${file_path}_pred.nii.gz -bin 0.5 -o ${file_path}_pred_bin.nii.gz

  # Compute CSA
  sct_process_segmentation -i ${file_path}_pred_bin.nii.gz-vert 2:3 -vertfile ${file_seg_labeled}.nii.gz -o ${PATH_RESULTS}/csa_pred_${contrast_seg}.csv -append 1

done
# ------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------

# Display results (to easily compare integrity across SCT versions)
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"