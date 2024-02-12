#!/bin/bash
#
# Compare the CSA of soft GT thresholded at different values on spine-generic test dataset.
# 
# Adapted from: https://github.com/ivadomed/model_seg_sci/blob/main/baselines/comparison_with_other_methods_sc.sh
#
# Usage:
#     sct_run_batch -config config.json
#
# Example of config.json:
# {
#  "path_data"   : "<PATH_TO_DATASET>",
#  "path_output" : "<PATH_TO_DATASET>_2023-08-18",
#  "script"      : "<PATH_TO_REPO>/model_seg_sci/baselines/comparison_with_other_methods.sh",
#  "jobs"        : 8,
#  "script_args" : "<PATH_TO_REPO>/model_seg_sci/packaging/run_inference_single_subject.py <PATH_TO_MODEL>/sci-multisite-model <PATH_TO_CONTRAST-AGNOSTIC_REPO>/monai/run_inference_single_image.py <PATH_TO_CONTRAST-AGNOSTIC_MODEL>"
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

# # Uncomment for full verbose
# set -x

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
QC_DATASET=$2           # dataset name to generate QC for
PATH_NNUNET_SCRIPT=$3   # path to the nnUNet contrast-agnostic run_inference_single_subject.py
PATH_NNUNET_MODEL=$4    # path to the nnUNet contrast-agnostic model
# PATH_MONAI_SCRIPT=$3    # path to the MONAI contrast-agnostic run_inference_single_subject.py
# PATH_MONAI_MODEL_SOFT=$4     # path to the MONAI contrast-agnostic model trained on soft labels
# PATH_MONAI_MODEL_SOFTBIN=$5     # path to the MONAI contrast-agnostic model trained on soft_bin labels

echo "SUBJECT: ${SUBJECT}"
echo "QC_DATASET: ${QC_DATASET}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"
# echo "PATH_MONAI_SCRIPT: ${PATH_MONAI_SCRIPT}"
# echo "PATH_MONAI_MODEL_SOFT: ${PATH_MONAI_MODEL_SOFT}"
# echo "PATH_MONAI_MODEL_SOFTBIN: ${PATH_MONAI_MODEL_SOFTBIN}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------

# Get ANIMA binaries path
anima_binaries_path=$(grep "^anima = " ~/.anima/config.txt | sed "s/.* = //" | sed 's/\/$//')

# Compute ANIMA segmentation performance metrics
compute_anima_metrics(){
  local FILESEG="$1"
  local FILEGT="$2"

  # We have to copy qform matrix from seg-manual to the automatically generated segmentation to avoid ITK error:
  # "Description: ITK ERROR: SegmentationMeasuresImageFilter(): Inputs do not occupy the same physical space!"
  # Related to the following issue : https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4135
  sct_image -i ${FILEGT}.nii.gz -copy-header ${FILESEG}.nii.gz -o ${FILESEG}_updated_header.nii.gz

  # Compute ANIMA segmentation performance metrics
  # -i : input segmentation
  # -r : GT segmentation
  # -o : output file
  # -d : surface distances evaluation
  # -s : compute metrics to evaluate a segmentation
  # -X : stores results into a xml file.
  ${anima_binaries_path}/animaSegPerfAnalyzer -i ${FILESEG}_updated_header.nii.gz -r ${FILEGT}.nii.gz -o ${PATH_RESULTS}/${FILESEG} -d -s -X

  rm ${FILESEG}_updated_header.nii.gz
}

# Copy GT segmentation (located under derivatives/labels)
copy_gt_seg(){
  local file="$1"
  local label_suffix="$2"
  # Construct file name to GT segmentation located under derivatives/labels
  FILESEG="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_${label_suffix}.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILESEG"
  if [[ -e $FILESEG ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILESEG ${file}_seg-manual.nii.gz
  else
      echo "File ${FILESEG}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual Segmentation ${FILESEG} does not exist. Exiting."
      exit 1
  fi
}

# TODO: Fix the contrast input for deepseg and propseg (i.e. dwi, mton, mtoff won't work)
# Segment spinal cord using methods available in SCT (sct_deepseg_sc or sct_propseg), resample the prediction back to
# native resolution and compute CSA in native space
segment_sc() {
  local file="$1"
  local contrast="$2"
  local method="$3"     # deepseg or propseg
  local kernel="$4"     # 2d or 3d; only relevant for deepseg
  local file_gt_vert_label="$5"
  local native_res="$6"

  # Segment spinal cord
  if [[ $method == 'deepseg' ]];then
      FILESEG="${file}_seg_${method}_${kernel}"

      # Get the start time
      start_time=$(date +%s)
      # Run SC segmentation
      sct_deepseg_sc -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -kernel ${kernel} -qc ${PATH_QC} -qc-subject ${SUBJECT}
      # Get the end time
      end_time=$(date +%s)
      # Calculate the time difference
      execution_time=$(python3 -c "print($end_time - $start_time)")
      echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv
  
  elif [[ $method == 'propseg' ]]; then
      FILESEG="${file}_seg_${method}"

      # Get the start time
      start_time=$(date +%s)
      # Run SC segmentation
      sct_propseg -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
      # Get the end time
      end_time=$(date +%s)
      # Calculate the time difference
      execution_time=$(python3 -c "print($end_time - $start_time)")
      echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

      # Remove centerline (we don't need it)
      rm ${file}_centerline.nii.gz

  fi

  # Compute CSA from the the SC segmentation resampled back to native resolution using the GT vertebral labels
  sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:4 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/csa_label_types_c24.csv -append 1

}

# Segment spinal cord using the contrast-agnostic nnUNet model
segment_sc_nnUNet(){
  local file="$1"
  local kernel="$2"     # 2d or 3d

  # FILESEG="${file}_seg_nnunet_${kernel}"
  FILESEG="${file}_seg_nnunet"

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILESEG}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainer__nnUNetPlans__${kernel} -pred-type sc -use-gpu
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Generate QC report
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # # Compute CSA from the prediction resampled back to native resolution using the GT vertebral labels
  # sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:4 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/csa_label_types_c24.csv -append 1

}

# Segment spinal cord using the MONAI contrast-agnostic model
segment_sc_MONAI(){
  local file="$1"
  local label_type="$2"     # soft or soft_bin

	if [[ $label_type == 'soft' ]]; then
		FILEPRED="${file}_seg_monai_soft"
		PATH_MONAI_MODEL=${PATH_MONAI_MODEL_SOFT}
	
	elif [[ $label_type == 'soft_bin' ]]; then
    FILEPRED="${file}_seg_monai_bin"
		PATH_MONAI_MODEL=${PATH_MONAI_MODEL_SOFTBIN}
	
	fi

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MONAI_MODEL} --device gpu
  # Rename MONAI output
  mv ${file}_pred.nii.gz ${FILEPRED}.nii.gz
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILEPRED},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Binarize MONAI output (which is soft by default); output is overwritten
  sct_maths -i ${FILEPRED}.nii.gz -bin 0.5 -o ${FILEPRED}.nii.gz

  # Generate QC report with soft prediction
  sct_qc -i ${file}.nii.gz -s ${FILEPRED}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # compute ANIMA metrics
  compute_anima_metrics ${FILEPRED} ${file}_seg-manual

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

# Note: we use '/./' in order to include the sub-folder 'ses-0X'
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
if [[ $QC_DATASET == "sci-colorado" ]]; then
  contrast="T2w"
  label_suffix="seg-manual"

elif [[ $QC_DATASET == "basel-mp2rage-rpi" ]]; then
  contrast="UNIT1"
  label_suffix="label-SC_seg"

elif [[ $QC_DATASET == "dcm-zurich" ]]; then
  contrast="acq-axial_T2w"
  label_suffix="label-SC_mask-manual"

elif [[ $QC_DATASET == "stanford-epi" ]]; then
  contrast="task-rest_desc-mocomean_bold"
  label_suffix="spinalcord_mask"
fi
# TODO: add stanford EPI data for QC

echo "Contrast: ${contrast}"

# Copy source images
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}*${contrast}.* .

# Go to the folder where the data is
cd ${PATH_DATA_PROCESSED}/${SUBJECT}/anat

# Get file name
file="${SUBJECT}_${contrast}"

# Check if file exists
if [[ ! -e ${file}.nii.gz ]]; then
    echo "File ${file}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file}.nii.gz does not exist. Exiting."
    exit 1
fi

# Copy GT spinal cord segmentation
copy_gt_seg "${file}" "${label_suffix}"

# Segment SC using different methods, binarize at 0.5 and compute QC
# segment_sc_MONAI ${file} 'soft'
# segment_sc_MONAI ${file} 'soft_bin'

segment_sc_nnUNet ${file} '3d_fullres'
# # TODO: run on deep/progseg after fixing the contrasts for those
# segment_sc ${file_res} 't2' 'deepseg' '2d' "${file}_seg-manual" ${native_res}
# segment_sc ${file_res} 't2' 'propseg' '' "${file}_seg-manual" ${native_res}


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
