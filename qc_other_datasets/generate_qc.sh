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
QC_DATASET=$2           # dataset name to generate QC for
PATH_NNUNET_SCRIPT=$3   # path to the nnUNet contrast-agnostic run_inference_single_subject.py
PATH_NNUNET_MODEL=$4    # path to the nnUNet contrast-agnostic model
PATH_MONAI_SCRIPT=$5    # path to the MONAI contrast-agnostic run_inference_single_subject.py
# PATH_SWIN_MODEL=$7
# PATH_MEDNEXT_MODEL=$8
PATH_MONAI_MODEL_1=$6     
PATH_MONAI_MODEL_2=$7     
PATH_MONAI_MODEL_3=$8     

echo "SUBJECT: ${SUBJECT}"
echo "QC_DATASET: ${QC_DATASET}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"
echo "PATH_MONAI_SCRIPT: ${PATH_MONAI_SCRIPT}"
echo "PATH_MONAI_MODEL_1: ${PATH_MONAI_MODEL_1}"
echo "PATH_MONAI_MODEL_2: ${PATH_MONAI_MODEL_2}"
echo "PATH_MONAI_MODEL_3: ${PATH_MONAI_MODEL_3}"
# echo "PATH_SWIN_MODEL: ${PATH_SWIN_MODEL}"
# echo "PATH_MEDNEXT_MODEL: ${PATH_MEDNEXT_MODEL}"

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
      # rsync -avzh ${FILESEG/.nii.gz/.json} ${file}_seg-manual.json
  else
      echo "File ${FILESEG}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual Segmentation ${FILESEG} does not exist. Exiting."
      exit 1
  fi
}

# Segment spinal cord using methods available in SCT (sct_deepseg_sc or sct_propseg), resample the prediction back to
# native resolution and compute CSA in native space
segment_sc() {
  local file="$1"
  local method="$2"     # deepseg or propseg
  local contrast="$3"   # used for input arg `-c`
  local kernel="2d"     # 2d or 3d; only relevant for deepseg

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

  # # Compute CSA from the the SC segmentation resampled back to native resolution using the GT vertebral labels
  # sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:4 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/csa_label_types_c24.csv -append 1

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
	# 	FILEPRED="${file}_seg_monai_soft_input"
	# 	PATH_MODEL=${PATH_MONAI_MODEL_SOFT}
	
	# elif [[ $label_type == 'bin' ]]; then
  #   FILEPRED="${file}_seg_monai_bin_input"
	# 	PATH_MODEL=${PATH_MONAI_MODEL_BIN}
	
	# fi
	if [[ $model == 'v2x' ]]; then
		FILEPRED="${file}_seg_${model}"
		PATH_MODEL=${PATH_MONAI_MODEL_1}
    max_feat=320
  
  elif [[ $model == 'v2x_contour' ]]; then
    FILEPRED="${file}_seg_${model}"
    PATH_MODEL=${PATH_MONAI_MODEL_2}
    max_feat=384
  
  elif [[ $model == 'v2x_contour_dcm' ]]; then
    FILEPRED="${file}_seg_${model}"
    PATH_MODEL=${PATH_MONAI_MODEL_3}
    max_feat=384
	
	# elif [[ $model == 'swinunetr' ]]; then
  #   FILEPRED="${file}_seg_swinunetr"
  #   PATH_MODEL=${PATH_SWIN_MODEL}
  	
	fi

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  # python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MODEL} --device gpu --model ${model}
  python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MODEL} --device gpu --model monai --pred-type soft --pad-mode edge --max-feat ${max_feat}
  # Rename MONAI output
  mv ${file}_pred.nii.gz ${FILEPRED}.nii.gz
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILEPRED},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # # Generate QC report on soft predictions
  # sct_qc -i ${file}.nii.gz -s ${FILEPRED}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Binarize MONAI output (which is soft by default); output is overwritten
  sct_maths -i ${FILEPRED}.nii.gz -bin 0.5 -o ${FILEPRED}.nii.gz

  # Generate QC report 
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
  deepseg_input_c="t2"

elif [[ $QC_DATASET == "basel-mp2rage" ]]; then
  contrast="UNIT1"
  label_suffix="label-SC_seg"
  deepseg_input_c="t1"

elif [[ $QC_DATASET == "dcm-zurich" ]]; then
  contrast="acq-axial_T2w"
  label_suffix="label-SC_mask-manual"
  deepseg_input_c="t2"

elif [[ $QC_DATASET == "lumbar-epfl" ]]; then
  contrast="T2w"
  label_suffix="seg-manual"
  deepseg_input_c="t2"

elif [[ $QC_DATASET == "canproco" ]]; then
  contrast="PSIR"
  label_suffix="seg-manual"
  deepseg_input_c="t2"

else
  echo "ERROR: Dataset ${QC_DATASET} not recognized. Exiting."
  exit 1

fi

echo "Contrast: ${contrast}"

# Copy source images
# check if the file exists
# if [[ ! -e ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}*${contrast}.nii.gz ]]; then
#     echo "${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}*${contrast}.nii.gz"
#     echo "File ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}*${contrast}.* does not exist" >> ${PATH_LOG}/missing_files.log
#     echo "ERROR: File ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}*${contrast}.* does not exist. Exiting."
#     exit 1
# else 
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}*${contrast}.* .
# fi

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

# Generate QC report the GT spinal cord segmentation
sct_qc -i ${file}.nii.gz -s ${file}_seg-manual.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Segment SC using different methods, binarize at 0.5 and compute QC
CUDA_VISIBLE_DEVICES=2 segment_sc_MONAI ${file} 'v2x'
CUDA_VISIBLE_DEVICES=2 segment_sc_MONAI ${file} 'v2x_contour'
CUDA_VISIBLE_DEVICES=3 segment_sc_MONAI ${file} 'v2x_contour_dcm'
# segment_sc_MONAI ${file} 'monai'
# segment_sc_MONAI ${file} 'swinunetr'

# CUDA_VISIBLE_DEVICES=2 segment_sc_nnUNet ${file} '3d_fullres'
segment_sc ${file} 'deepseg' ${deepseg_input_c}

# Create new "_clean" folder with BIDS-updated derivatives filenames
date_time=$(date +"%Y-%m-%d %H:%M:%S")
json_dict='{
  "GeneratedBy": [
    {
      "Name": "contrast-agnostic-softseg-spinalcord",
      "Version": "2.3",
      "Date": "'$date_time'"
    }
  ]
}'

# PATH_DATA_PROCESSED_CLEAN="${PATH_DATA_PROCESSED}_clean"
# # create new folder and copy only the predictions
# mkdir -p ${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat

# rsync -avzh ${file}_seg_soft_monai_4datasets.nii.gz ${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat/${file%%_*}_space-other_${contrast}_desc-softseg_label-SC_seg.nii.gz
# rsync -avzh ${file}_seg-manual.json ${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat/${file%%_*}_space-other_${contrast}_desc-softseg_label-SC_seg.json

# # create json file
# echo $json_dict > ${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat/${file%%_*}_space-other_${contrast}_desc-softseg_label-SC_seg.json
# # re-save json files with indentation
# python -c "import json;
# json_file = '${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat/${file%%_*}_space-other_${contrast}_desc-softseg_label-SC_seg.json'
# with open(json_file, 'r') as f:
#     data = json.load(f)
#     json.dump(data, open(json_file, 'w'), indent=4)
# "

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
