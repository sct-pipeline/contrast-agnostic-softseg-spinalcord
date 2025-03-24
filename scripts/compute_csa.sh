#!/bin/bash
#
# Compute the CSA of a lifelong learning contrast-agnostic segmentation model on the spine-generic test dataset.
# To be used within the `compute_morphometrics_spine_generic.sh` script to run the CSA analysis on the latest version 
# of the contrast-agnostic model
#
# Author: Naga Karthik
#

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Print retrieved variables from the sct_run_batch script to the log (to allow easier debug)
echo "Retrieved variables from from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"

# Variable passed by `sct_run_batch -script-args`
SUBJECT=$1
MODEL_VERSION=$2
# CUDA_DEVICE=$2
PATH_NNUNET_SCRIPT=$3   # path to the nnUNet contrast-agnostic run_inference_single_subject.py
PATH_NNUNET_MODEL=$4

echo "SUBJECT: ${SUBJECT}"
echo "USING CUDA DEVICE ID: ${CUDA_DEVICE}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------

# Check if manual label already exists. If it does, copy it locally.
# NOTE: manual disc labels should go from C1-C2 to C7-T1.
label_vertebrae(){
  local file="$1"
  local contrast="$2"

  # Update global variable with segmentation file name
  FILESEG="${file}_softseg_bin"
  FILELABEL="${file}_discs"

  # Get vertebral levels by projecting discs on the spinal cord segmentation
  # Note: we are using sct_label_utils over sct_label_vertebrae here to avoid straightening (which takes a lot of time)
  sct_label_utils -i ${FILESEG}.nii.gz -disc ${FILELABEL}.nii.gz -o ${FILESEG}_labeled.nii.gz
}


# Copy GT spinal cord disc labels (located under derivatives/labels)
copy_gt_disc_labels(){
  local file="$1"
  local type="$2"
  local contrast="$3"

  if [[ $contrast == "T1w" ]] || [[ $contrast == "T2w" ]]; then
    file_name="${file%%_*}_${contrast}_label-discs_dlabel"
  elif [[ $contrast == "T2star" ]]; then
    file_name="${file%%_*}_${contrast}_label-discs_desc-warp_dlabel"
  elif [[ $contrast == "MTon" ]]; then
    file_name="${file%%_*}_flip-1_mt-on_MTS_label-discs_desc-warp_dlabel"
  elif [[ $contrast == "MToff" ]]; then
    file_name="${file%%_*}_flip-2_mt-off_MTS_label-discs_desc-warp_dlabel"
  elif [[ $contrast == "DWI" ]]; then
    file_name="${file%%_*}_rec-average_dwi_label-discs_desc-warp_dlabel"
  fi
  # Construct file name to GT segmentation located under derivatives/labels
  FILEDISCLABELS="${PATH_DATA}/derivatives/labels/${SUBJECT}/${type}/${file_name}.nii.gz"
  echo ""
  echo "Looking for manual disc labels: $FILEDISCLABELS"
  if [[ -e $FILEDISCLABELS ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILEDISCLABELS ${file}_discs.nii.gz
  else
      echo "File ${FILEDISCLABELS} does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual Disc Labels ${FILEDISCLABELS} does not exist. Exiting."
      exit 1
  fi
}


# Copy GT soft binarized segmentation (located under derivatives/labels_softseg_bin)
copy_gt_softseg_bin(){
  local file="$1"
  local type="$2"
  # Construct file name to GT segmentation located under derivatives/labels_softseg_bin
  # NOTE: the naming conventions are in the revised BIDS format
  FILESEG="${PATH_DATA}/derivatives/labels_softseg_bin/${SUBJECT}/${type}/${file}_desc-softseg_label-SC_seg.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILESEG"
  if [[ -e $FILESEG ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILESEG ${file}_softseg_bin.nii.gz
  else
      echo "File ${FILESEG} does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual Segmentation ${FILESEG} does not exist. Exiting."
      exit 1
  fi
}


# Segment spinal cord
segment_sc(){
  local file="$1"
  local file_gt_vert_label="$2"
  local model_basename="$3"     # 2d or 3d
  local contrast="$4"   # used only for saving output file name
  # local kernel="$5"    # 2d or 3d_fullres

  FILESEG="${file%%_*}_${contrast}_seg_${model_basename}"

  # Get the start time
  start_time=$(date +%s)
  # # Run SC segmentation
  # python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILESEG}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainer__nnUNetPlans__${kernel} -pred-type sc -use-gpu -use-best-checkpoint
  # Run SC segmentation (natively with sct_deepseg)
  sct_deepseg -task seg_sc_contrast_agnostic -i ${file}.nii.gz -o ${FILESEG}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # # Generate QC report
  # sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Compute CSA averaged across all slices C2-C3 vertebral levels for plotting the STD across contrasts
  # NOTE: this is per-level because not all contrasts have thes same FoV (C2-C3 is what all contrasts have in common)
  sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:3 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/csa_c2c3.csv -append 1

}


# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# get starting time:
start=`date +%s`

# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/* .
# copy DWI data
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/dwi/* .

# ------------------------------------------------------------------------------
# DEFINE CONTRASTS
# ------------------------------------------------------------------------------
# contrasts="T1w T2w T2star flip-1_mt-on_MTS flip-2_mt-off_MTS rec-average_dwi"
contrasts="rec-average_dwi"

# Loop across contrasts
for contrast in ${contrasts}; do

  if [[ $contrast == "rec-average_dwi" ]]; then
    type="dwi"
  else
    type="anat"
  fi

  # go to the folder where the data is
  cd ${PATH_DATA_PROCESSED}/${SUBJECT}/${type}

  # Get file name
  file="${SUBJECT}_${contrast}"

  # Check if file exists
  if [[ ! -e ${file}.nii.gz ]]; then
      echo "File ${file}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: File ${file}.nii.gz does not exist. Exiting."
      exit 1
  fi

  # rename contrasts
  if [[ $contrast == "flip-1_mt-on_MTS" ]]; then
    contrast="MTon"
  elif [[ $contrast == "flip-2_mt-off_MTS" ]]; then
    contrast="MToff"
  elif [[ $contrast == "rec-average_dwi" ]]; then
    contrast="DWI"
  fi

  # ------------------------------------------------------------------------------
  # COMPUTE CSA OF GT MASKS
  # ------------------------------------------------------------------------------
  # # Copy GT spinal cord segmentation
  # copy_gt_seg "${file}" "${type}"

  # Copy soft GT spinal cord segmentation
  copy_gt_softseg_bin "${file}" "${type}"

  # Copy GT disc labels segmentation
  copy_gt_disc_labels "${file}" "${type}" "${contrast}"

  # Label vertebral levels in the native resolution
  label_vertebrae ${file} 't2'

  # Rename the softseg_bin GT with the shorter contrast name
  FILEBIN="${file%%_*}_${contrast}_softseg_bin"
  if [[ "${file}_softseg_bin.nii.gz" != "${FILEBIN}.nii.gz" ]]; then
    mv ${file}_softseg_bin.nii.gz ${FILEBIN}.nii.gz
  fi

  # Generate QC report 
  sct_qc -i ${file}.nii.gz -s ${FILEBIN}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Compute CSA averaged across all slices C2-C3 vertebral levels for plotting the STD across contrasts
  # NOTE: this is per-level because not all contrasts have thes same FoV (C2-C3 is what all contrasts have in common)
  sct_process_segmentation -i ${FILEBIN}.nii.gz -vert 2:3 -vertfile ${file}_softseg_bin_labeled.nii.gz -o $PATH_RESULTS/csa_c2c3.csv -append 1

  # ------------------------------------------------------------------------------
  # COMPUTE CSA OF AUTOMATIC PREDICTIONS
  # ------------------------------------------------------------------------------
  # Segment SC (i.e. run inference) and compute CSA
  # model_name=$(basename ${PATH_NNUNET_MODEL})
  # CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} segment_sc_nnUNet ${file} "${file}_softseg_bin" ${model_name} ${contrast} '3d_fullres'
  segment_sc ${file} "${file}_softseg_bin" ${MODEL_VERSION} ${contrast}

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
