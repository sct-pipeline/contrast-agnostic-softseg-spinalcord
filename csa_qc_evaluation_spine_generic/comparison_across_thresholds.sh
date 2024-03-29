#!/bin/bash
#
# Compare the contrast-agnostic model (MONAI and nnUNet versions) with other methods (sct_propseg, sct_deepseg_sc)
# across different resolutions on spine-generic test dataset.
# 
# Adapted from: https://github.com/ivadomed/model_seg_sci/blob/main/baselines/comparison_with_other_methods_sc.sh
#
# Usage:
#     sct_run_batch -config config.json
#
# Example of config.json:
# {
#  "path_data"   : "<PATH_TO_DATASET>",
#  "path_output" : "<PATH_TO_DATASET>/results_csa/across_thresholds/csa_softbin_thr_20240218",
#  "script"      : "<PATH_TO_REPO>/csa_qc_evaluation_spine_generic/comparison_across_thresholds.sh",
#  "jobs"        : 5,
#  "script_args" : "<PATH_TO_REPO>/monai/run_inference_single_image.py <PATH_TO_MONAI_MODEL>"
# }
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Author: Jan Valosek and Naga Karthik
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

SUBJECT=$1
PATH_MONAI_SCRIPT=$2    # path to the MONAI contrast-agnostic run_inference_single_subject.py
PATH_MONAI_MODEL=$3     # path to the MONAI contrast-agnostic model

echo "SUBJECT: ${SUBJECT}"
echo "PATH_MONAI_SCRIPT: ${PATH_MONAI_SCRIPT}"
echo "PATH_MONAI_MODEL: ${PATH_MONAI_MODEL}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------

# Check if manual label already exists. If it does, copy it locally.
# NOTE: manual disc labels should go from C1-C2 to C7-T1.
label_vertebrae(){
  local file="$1"
  local contrast="$2"
  
  # Update global variable with segmentation file name
  FILESEG="${file}_seg-manual"
  FILELABEL="${file}_discs"

  # Label vertebral levels
  sct_label_utils -i ${file}.nii.gz -disc ${FILELABEL}.nii.gz -o ${FILESEG}_labeled.nii.gz

  # # Run QC
  # sct_qc -i ${file}.nii.gz -s ${file_seg}_labeled.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}
}


# Segment spinal cord using the MONAI contrast-agnostic model, resample the prediction back to native resolution and
# compute CSA in native space
segment_sc_MONAI(){
  local file="$1"
  local file_gt_vert_label="$2"
  local threshold="$3"
  local contrast="$4"

  # Remove the 0. for the file name
  thr_e="$(echo $thr | awk -F'.' '{print $2}')"

  # keep on the subject and the re-named contrast in output file name
  FILESEG="${file%%_*}_${contrast}_seg_monai_thr_${thr_e}"

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MONAI_MODEL} --device gpu --pred-thr ${threshold} --keep-largest
  # Rename MONAI output
  mv ${file}_pred.nii.gz ${FILESEG}.nii.gz
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # # Generate QC report with soft prediction
  # sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Compute CSA from the soft prediction resampled back to native resolution using the GT vertebral labels
  sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:4 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/csa_softbin_thresh_c24.csv -append 1

}

# Copy GT spinal cord disc labels (located under derivatives/labels)
copy_gt_disc_labels(){
  local file="$1"
  local type="$2"
  # Construct file name to GT segmentation located under derivatives/labels
  FILEDISCLABELS="${PATH_DATA}/derivatives/labels/${SUBJECT}/${type}/${file}_discs.nii.gz"
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

# Copy GT segmentation (located under derivatives/labels)
copy_gt_seg(){
  local file="$1"
  # Construct file name to GT segmentation located under derivatives/labels
  FILESEG="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_seg-manual.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILESEG"
  if [[ -e $FILESEG ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILESEG ${file}_seg-manual.nii.gz
  else
      echo "File ${FILESEG}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual Segmentation ${FILESEG}.nii.gz does not exist. Exiting."
      exit 1
  fi
}

# Copy GT soft segmentation (located under derivatives/labels_softseg)
copy_gt_softseg(){
  local file="$1"
  local type="$2"
  # Construct file name to GT segmentation located under derivatives/labels
  FILESEG="${PATH_DATA}/derivatives/labels_softseg/${SUBJECT}/${type}/${file}_softseg.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILESEG"
  if [[ -e $FILESEG ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILESEG ${file}_softseg.nii.gz
  else
      echo "File ${FILESEG} does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual Segmentation ${FILESEG} does not exist. Exiting."
      exit 1
  fi
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
# contrast
# ------------------------------------------------------------------------------
contrasts="T1w T2w T2star flip-1_mt-on_MTS flip-2_mt-off_MTS rec-average_dwi"
# contrasts="flip-2_mt-off_MTS rec-average_dwi"

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

  # Copy GT disc labels
  copy_gt_disc_labels "${file}" "${type}"

  # Copy GT spinal cord segmentation
  copy_gt_softseg "${file}" "${type}"

  # Label vertebral levels in the native resolution
  label_vertebrae ${file} 't2'

  # thresholds for thresholding the predictions
#   thresholds="0.2 0.15 0.1 0.05 0.01"
  thresholds="0.15 0.1 0.05 0.01 0.005"

  # rname contrast if flip 1 or flip 2
  if [[ $contrast == "flip-1_mt-on_MTS" ]]; then
    contrast="MTon"
  elif [[ $contrast == "flip-2_mt-off_MTS" ]]; then
    contrast="MToff"
  elif [[ $contrast == "rec-average_dwi" ]]; then
    contrast="DWI"
  fi

  # Loop across thresholds
  for thr in ${thresholds}; do

    echo "Thresholding the soft GT with threshold: ${thr} ..."

    # Remove the 0. for the file name
    thr_e="$(echo $thr | awk -F'.' '{print $2}')"

    # Threshold the soft GT 
    FILETHRESH="${file%%_*}_${contrast}_softseg_thr_${thr_e}"
    sct_maths -i ${file}_softseg.nii.gz -thr ${thr} -o ${FILETHRESH}.nii.gz

    # Compute the CSA of the thresholded soft GT
    sct_process_segmentation -i ${FILETHRESH}.nii.gz -vert 2:4 -vertfile ${file}_seg-manual_labeled.nii.gz -o $PATH_RESULTS/csa_softbin_thresh_c24.csv -append 1

    echo "Thresholding MONAI predictions with threshold: ${thr} ..."

    # Segment SC using different thresholds and compute CSA
    segment_sc_MONAI ${file} "${file}_seg-manual" ${thr} ${contrast}

  done

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
