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
#  "path_output" : "<PATH_TO_DATASET>/results_qc_other_datasets/qc-reports",
#  "script"      : "<PATH_TO_REPO>/qc_other_datasets/generate_qc.sh",
#  "jobs"        : 5,
#  "script_args" : "<DATASET_TO_QC> <PATH_TO_REPO>/nnUnet/run_inference_single_subject.py <PATH_TO_NNUNET_MODEL> <PATH_TO_REPO>/monai/run_inference_single_image.py <PATH_TO_MONAI_MODEL> <PATH_TO_SWINUNETR_MODEL> <PATH_TO_MEDNEXT_MODEL>"
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
PATH_MONAI_SCRIPT=$2    # path to the MONAI contrast-agnostic run_inference_single_subject.py
PATH_MONAI_MODEL_OLD=$3     # path to the MONAI contrast-agnostic model trained on soft bin labels
PATH_MONAI_MODEL_NEW=$4

echo "SUBJECT: ${SUBJECT}"
echo "PATH_MONAI_SCRIPT: ${PATH_MONAI_SCRIPT}"
echo "PATH_MONAI_MODEL_OLD: ${PATH_MONAI_MODEL_OLD}"
echo "PATH_MONAI_MODEL_NEW: ${PATH_MONAI_MODEL_NEW}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------

# Get ANIMA binaries path
anima_binaries_path=$(grep "^anima = " ~/.anima/config.txt | sed "s/.* = //" | sed 's/\/$//')

# Compute ANIMA segmentation performance metrics
compute_anima_metrics(){
  local FILESEG="$1"
  local FILEGT="$2"

  # # We have to copy qform matrix from seg-manual to the automatically generated segmentation to avoid ITK error:
  # # "Description: ITK ERROR: SegmentationMeasuresImageFilter(): Inputs do not occupy the same physical space!"
  # # Related to the following issue : https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4135
  sct_image -i ${FILEGT}.nii.gz -copy-header ${FILESEG}.nii.gz -o ${FILESEG}_updated_header.nii.gz

  # # print the header of the updated segmentation
  # sct_image -i ${FILESEG}_updated_header.nii.gz -header
  # echo ""
  # sct_image -i ${FILEGT}.nii.gz -header

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

# Copy GT soft binarized segmentation (located under derivatives/labels_softseg_bin)
copy_gt_softseg_bin(){
  local file="$1"
  local type="$2"
  # Construct file name to GT segmentation located under derivatives/labels_softseg_bin
  # NOTE: the naming conventions are in the revised BIDS format
  # FILESEG="${PATH_DATA}/derivatives/labels_softseg_bin/${SUBJECT}/${type}/${file}_desc-softseg_label-SC_seg.nii.gz"
  FILESEG="${PATH_DATA}/derivatives/labels_softseg/${SUBJECT}/${type}/${file}_label-SC_softseg.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILESEG"
  if [[ -e $FILESEG ]]; then
      echo "Found! Copying ..."
      # rsync -avzh $FILESEG ${file}_softseg_bin.nii.gz
      rsync -avzh $FILESEG ${file}_softseg.nii.gz
  else
      echo "File ${FILESEG} does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual Segmentation ${FILESEG} does not exist. Exiting."
      exit 1
  fi
}

# Copy GT soft segmentation (located under derivatives/labels_softseg)
copy_gt_softseg(){
  local file="$1"
  local type="$2"
  local location="$3"
  # Construct file name to GT segmentation located under derivatives/labels
  if [[ $location == 'duke' ]]; then
    FILESEG="${PATH_DATA}/derivatives/labels_softseg/${SUBJECT}/${type}/${file}_softseg.nii.gz"
  elif [[ $location == 'git-annex' ]]; then
    FILESEG="${PATH_DATA}/derivatives/labels_softseg/${SUBJECT}/${type}/${file}_label-SC_softseg.nii.gz"
  fi
  
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
  # local label_type="$2"     # soft or soft_bin
  local model="$2"     # monai, swinunetr, mednext

	# if [[ $label_type == 'soft' ]]; then
	# 	FILEPRED="${file}_seg_monai_soft"
	# 	PATH_MONAI_MODEL=${PATH_MONAI_MODEL_SOFT}
	
	# elif [[ $label_type == 'soft_bin' ]]; then
  #   FILEPRED="${file}_seg_monai_bin"
	# 	PATH_MONAI_MODEL=${PATH_MONAI_MODEL_SOFTBIN}
	
	# fi
	if [[ $model == 'monai_old' ]]; then
		FILEPRED="${file}_seg_monai_old"
		PATH_MODEL=${PATH_MONAI_MODEL_OLD}
	
	elif [[ $model == 'monai_new' ]]; then
    FILEPRED="${file}_seg_monai_new"
    PATH_MODEL=${PATH_MONAI_MODEL_NEW}
  	
	fi

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MODEL} --device gpu --model monai --pred-type soft
  # Rename MONAI output
  mv ${file}_pred.nii.gz ${FILEPRED}.nii.gz
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILEPRED},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Binarize MONAI output (which is soft by default); output is overwritten
  sct_maths -i ${FILEPRED}.nii.gz -bin 0.5 -o ${FILEPRED}.nii.gz

  # Generate QC report 
  sct_qc -i ${file}.nii.gz -s ${FILEPRED}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # compute ANIMA metrics
  compute_anima_metrics ${FILEPRED} ${file}_softseg_bin

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
rsync -Ravzh ${PATH_DATA}/derivatives/data_preprocessed/./${SUBJECT}/anat/* .
# copy DWI data
rsync -Ravzh ${PATH_DATA}/derivatives/data_preprocessed/./${SUBJECT}/dwi/* .

# # Copy source images
# # Note: we use '/./' in order to include the sub-folder 'ses-0X'
# # We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
# rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/* .
# # copy DWI data
# rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/dwi/* .


# ------------------------------------------------------------------------------
# contrast
# ------------------------------------------------------------------------------
contrasts="space-other_T1w space-other_T2w space-other_T2star flip-1_mt-on_space-other_MTS flip-2_mt-off_space-other_MTS rec-average_dwi"
# contrasts="space-other_T1w rec-average_dwi"
# contrasts="flip-1_mt-on_space-other_MTS flip-2_mt-off_space-other_MTS"
# contrasts="T1w T2w T2star flip-1_mt-on_MTS flip-2_mt-off_MTS rec-average_dwi"
# contrasts="flip-1_mt-on_MTS flip-2_mt-off_MTS"

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

  # # Copy GT spinal cord segmentation
  # copy_gt_softseg_bin "${file}" "${type}"

  # Copy GT spinal cord segmentation
  copy_gt_softseg "${file}" "${type}" 'git-annex'
  sct_maths -i ${file}_softseg.nii.gz -bin 0.5 -o ${file}_softseg_bin.nii.gz

  # reorient the GT segmentation to RPI
  sct_image -i ${file}_softseg_bin.nii.gz -setorient RPI -o ${file}_softseg_bin.nii.gz
  
  # reorient the image to RPI
  sct_image -i ${file}.nii.gz -setorient RPI -o ${file}.nii.gz

  # Generate QC report for the GT segmentation
  sct_qc -i ${file}.nii.gz -s ${file}_softseg_bin.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Segment SC using different methods, binarize at 0.5 and compute QC
  # segment_sc_MONAI ${file} 'soft'
  # segment_sc_MONAI ${file} 'soft_bin'
  # CUDA_VISIBLE_DEVICES=1 segment_sc_MONAI ${file} 'monai_old'
  CUDA_VISIBLE_DEVICES=3 segment_sc_MONAI ${file} 'monai_new'

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
