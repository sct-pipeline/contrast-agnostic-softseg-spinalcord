#!/bin/bash
#
# Process data. Note. Input data are already resampled and reoriented to RPI.
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
EXCLUDE_LIST=$2

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

segment_if_does_not_exist(){
  local file="$1"
  local contrast="$2"
  # Find contrast
  if [[ $contrast == "dwi" ]]; then
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
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg_sc -i ${file}.nii.gz -c $contrast -qc ${PATH_QC} -qc-subject ${SUBJECT}
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

yaml() {
    python3 -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
}

# SCRIPT STARTS HERE
# ==============================================================================

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

cd ${SUBJECT}

# Check if a list of images to exclude was passed.
if [ -z "$EXCLUDE_LIST" ]; then
  EXCLUDE=""
else
  EXCLUDE=$(yaml ${EXCLUDE_LIST} "['FILES_SEG']")
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
# Go to anat folder where all structural data are located
cd ./anat/

# TODO: put find seg-manual in the loop when all segmentations are in the derivatives
# T1w
# ------------------------------------------------------------------------------

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t1 "t1"
file_t1_seg=$FILESEG

# T2w
# ------------------------------------------------------------------------------

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t2 "t2"
file_t2_seg=$FILESEG

# T2s
# ------------------------------------------------------------------------------

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t2s "t2s" 
file_t2s_seg=$FILESEG

# MTS
# ------------------------------------------------------------------------------

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t1w "t1" 
file_t1w_seg=$FILESEG
segment_if_does_not_exist $file_mton "t2s"
file_mton_seg=$FILESEG


# DWI
# ------------------------------------------------------------------------------
cd ../dwi
file_dwi_mean="${SUBJECT}_rec-average_dwi"

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist ${file_dwi_mean} "dwi"
file_dwi_mean_seg=$FILESEG

cd ..

# Create mask for regsitration
file_t2_mask="${file_t2_seg}_mask"
sct_create_mask -i ./anat/${file_t2}.nii.gz -p centerline,./anat/${file_t2_seg}.nii.gz -size 55mm -o ./anat/${file_t2_mask}.nii.gz 

# Loop through available contrasts
for file_path in "${inc_contrasts[@]}";do
  type=$(find_contrast $file_path)
    # Registration
    # ------------------------------------------------------------------------------
    file=${file_path/#"$type"}
    fileseg=${file_path}_seg
    sct_register_multimodal -i ${file_path}.nii.gz -d ./anat/${file_t2}.nii.gz -iseg ${fileseg}.nii.gz -dseg ./anat/${file_t2_seg}.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares,iter=10,poly=2 -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${file_path}_reg.nii.gz
    warping_field=${type}warp_${file}2${file_t2}

    # Generate SC segmentation coverage and register to T2w
    # ------------------------------------------------------------------------------
    sct_create_mask -i ${file_path}.nii.gz -o ${file_path}_ones.nii.gz -size 500 -p centerline,${fileseg}.nii.gz
    # Bring coverage mask to T2w space
    sct_apply_transfo -i ${file_path}_ones.nii.gz -d ./anat/${file_t2}.nii.gz -w ${warping_field}.nii.gz -x linear -o ${file_path}_ones_reg.nii.gz
    # Bring SC segmentation to T2w space
    sct_apply_transfo -i ${fileseg}.nii.gz -d ./anat/${file_t2_seg}.nii.gz -w ${warping_field}.nii.gz -x linear -o ${fileseg}_reg.nii.gz
done

# Create coverage mask for T2w
sct_create_mask -i ./anat/${file_t2}.nii.gz -o ./anat/${file_t2}_ones.nii.gz -size 500 -p centerline,./anat/${file_t2_seg}.nii.gz

# Create soft SC segmentation
# ------------------------------------------------------------------------------
# Sum all coverage images
contrasts_coverage=${inc_contrasts[@]/%/"_ones_reg.nii.gz"}
sct_maths -i ./anat/${file_t2}_ones.nii.gz -add $(eval echo ${contrasts_coverage[@]}) -o sum_coverage.nii.gz

# Sum all segmentations
contrasts_seg=${inc_contrasts[@]/%/"_seg_reg.nii.gz"}
sct_maths -i ./anat/${file_t2_seg}.nii.gz -add $(eval echo ${contrasts_seg[@]}) -o sum_sc_seg.nii.gz

# Divide sum_sc_seg by sum_coverage
sct_maths -i sum_sc_seg.nii.gz -div sum_coverage.nii.gz -o ./anat/${file_t2}_seg_soft.nii.gz

file_softseg=./anat/"${file_t2}_seg_soft"

# Create QC report of softseg on T2w
sct_qc -i ./anat/${file_t2}.nii.gz -s ${file_softseg}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Bring back softseg to native space and generate QC report 
# ------------------------------------------------------------------------------
for file_path in "${inc_contrasts[@]}";do
  type=$(find_contrast $file_path)
    file=${file_path/#"$type"}
    fileseg=${file_path}_seg
    warping_field_inv=${type}warp_${file_t2}2${file}

    # Bring softseg to native space
    sct_apply_transfo -i ${file_softseg}.nii.gz -d ${file_path}.nii.gz -w ${warping_field_inv}.nii.gz -x linear -o ${file_path}_softseg.nii.gz
    # Generate QC report
    sct_qc -i ${file_path}.nii.gz -s ${file_path}_softseg.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
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
