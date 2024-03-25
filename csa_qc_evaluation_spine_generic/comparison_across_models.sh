#!/bin/bash
#
# Compare the CSA of different models on the spine-generic test dataset
# 
# Adapted from: https://github.com/ivadomed/model_seg_sci/blob/main/baselines/comparison_with_other_methods_sc.sh
#
# Usage:
#     sct_run_batch -config config.json
#
# Example of config.json:
# {
#  "path_data"   : "<PATH_TO_DATASET>",
#  "path_output" : "<PATH_TO_DATASET>/results_csa/across_models/csa_c2c3_20240226",
#  "script"      : "<PATH_TO_REPO>/csa_qc_evaluation_spine_generic/comparison_across_models.sh",
#  "jobs"        : 5,
#  "script_args" : "<PATH_TO_REPO>/nnUnet/run_inference_single_subject.py <PATH_TO_NNUNET_MODEL> <PATH_TO_REPO>/monai/run_inference_single_image.py <PATH_TO_MONAI_MODEL> <PATH_TO_SWINUNETR_MODEL> <PATH_TO_MEDNEXT_MODEL>"
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
PATH_NNUNET_SCRIPT=$2   # path to the nnUNet contrast-agnostic run_inference_single_subject.py
PATH_NNUNET_MODEL=$3    # path to the nnUNet contrast-agnostic model
PATH_MONAI_SCRIPT=$4    # path to the MONAI contrast-agnostic run_inference_single_subject.py
PATH_MONAI_MODEL=$5     # path to the MONAI contrast-agnostic model trained on soft bin labels
PATH_MEDNEXT_MODEL=$6
PATH_UNETR_MODEL=$7
PATH_SWIN_MODEL=$8
PATH_SWIN_PTR_MODEL=$9

echo "SUBJECT: ${SUBJECT}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"
echo "PATH_MONAI_SCRIPT: ${PATH_MONAI_SCRIPT}"
echo "PATH_MONAI_MODEL: ${PATH_MONAI_MODEL}"
echo "PATH_MEDNEXT_MODEL: ${PATH_MEDNEXT_MODEL}"
echo "PATH_UNETR_MODEL: ${PATH_UNETR_MODEL}"
echo "PATH_SWIN_MODEL: ${PATH_SWIN_MODEL}"
echo "PATH_SWIN_PTR_MODEL: ${PATH_SWIN_PTR_MODEL}"

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
  local type="$2"
  # Construct file name to GT segmentation located under derivatives/labels
  FILESEG="${PATH_DATA}/derivatives/labels/${SUBJECT}/${type}/${file}_seg-manual.nii.gz"
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


# TODO: Fix the contrast input for deepseg and propseg (i.e. dwi, mton, mtoff won't work)
# Segment spinal cord using methods available in SCT (sct_deepseg_sc or sct_propseg), resample the prediction back to
# native resolution and compute CSA in native space
segment_sc() {
  local file="$1"
  local file_gt_vert_label="$2"
  local method="$3"     # deepseg or propseg
  local contrast_input="$4"   # used for input arg `-c`
  local contrast_name="$5"   # used only for saving output file name
  local csv_fname="$6"   # used for saving output file name

  # Segment spinal cord
  if [[ $method == 'deepseg' ]];then
      # FILESEG="${file}_seg_${method}_${kernel}"
      FILESEG="${file%%_*}_${contrast_name}_seg_${method}_2d"

      # Get the start time
      start_time=$(date +%s)
      # Run SC segmentation
      sct_deepseg_sc -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast_input} -kernel 2d -qc ${PATH_QC} -qc-subject ${SUBJECT}
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
  sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:3 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}.csv -append 1

}

# Segment spinal cord using the contrast-agnostic nnUNet model
segment_sc_nnUNet(){
  local file="$1"
  local file_gt_vert_label="$2"
  local kernel="$3"     # 2d or 3d
  local contrast="$4"   # used only for saving output file name
  local csv_fname="$5"   # used for saving output file name

  FILESEG="${file%%_*}_${contrast}_seg_nnunet"

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILESEG}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainer__nnUNetPlans__${kernel} -pred-type sc -use-gpu
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # # Generate QC report
  # sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Compute CSA from the prediction resampled back to native resolution using the GT vertebral labels
  sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:3 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}.csv -append 1

}

# Segment spinal cord using the MONAI contrast-agnostic model
segment_sc_MONAI(){
  local file="$1"
  local file_gt_vert_label="$2"
  local model="$3"     # monai or swinunetr or mednext
  local contrast="$4"   # used only for saving output file name
  local csv_fname="$5"   # used for saving output file name

	if [[ $model == 'monai' ]]; then
		FILESEG="${file%%_*}_${contrast}_seg_monai"
		PATH_MODEL=${PATH_MONAI_MODEL}
  
  elif [[ $model == 'unetr' ]]; then
    FILESEG="${file%%_*}_${contrast}_seg_unetr"
    PATH_MODEL=${PATH_UNETR_MODEL}
	
	elif [[ $model == 'swinunetr' ]]; then
    FILESEG="${file%%_*}_${contrast}_seg_swinunetr"
    PATH_MODEL=${PATH_SWIN_MODEL}

	elif [[ $model == 'swinpretrained' ]]; then
    FILESEG="${file%%_*}_${contrast}_seg_swinpretrained"
    PATH_MODEL=${PATH_SWIN_PTR_MODEL}  

  elif [[ $model == 'mednext' ]]; then
    FILESEG="${file%%_*}_${contrast}_seg_mednext"
    PATH_MODEL=${PATH_MEDNEXT_MODEL}
	
	fi

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  if [[ $model == 'swinpretrained' ]]; then
    # NOTE: pre-trained swinunetr model used different input/sliding_window_sizes, hence providing the -crop arg only for this model
    python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MODEL} --device gpu --model ${model} --pred-type soft -crop 64x160x-1
  else
    python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MODEL} --device gpu --model ${model} --pred-type soft
  fi
  # Rename MONAI output
  mv ${file}_pred.nii.gz ${FILESEG}.nii.gz
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Binarize MONAI output (which is soft by default); output is overwritten
  sct_maths -i ${FILESEG}.nii.gz -bin 0.5 -o ${FILESEG}.nii.gz

  # Generate QC report with soft prediction
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Compute CSA from the soft prediction resampled back to native resolution using the GT vertebral labels
  sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:3 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}.csv -append 1
}

# Ensemble the predictions from different models
segment_sc_ensemble(){
  local file="$1"
  local file_gt_vert_label="$2"
  local contrast="$3"   # used only for saving output file name
  local csv_fname="$4"   # used for saving output file name
  
  FILETEMP="${file%%_*}_${contrast}"
  
  FILESEG=${FILETEMP}_seg_ensemble

  # Get the start time
  start_time=$(date +%s)
  # Add segmentations from different models
  sct_maths -i ${FILETEMP}_seg_monai.nii.gz -add ${FILETEMP}_seg_nnunet.nii.gz ${FILETEMP}_seg_swinunetr.nii.gz ${FILETEMP}_seg_mednext.nii.gz -o ${FILESEG}.nii.gz
  # # Average the segmentations
  # sct_maths -i ${FILESEG}.nii.gz -div 4 -o ${FILESEG}.nii.gz
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Binarize MONAI output (which is soft by default); output is overwritten
  sct_maths -i ${FILESEG}.nii.gz -bin 0.5 -o ${FILESEG}.nii.gz

  # Generate QC report with soft prediction
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Compute CSA from the soft prediction resampled back to native resolution using the GT vertebral labels
  sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:3 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}.csv -append 1
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
# contrasts="T1w rec-average_dwi"

# output csv filename 
csv_fname="csa_models_c23"

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

  # Copy GT spinal cord segmentation
  copy_gt_seg "${file}" "${type}"

  # Copy soft GT spinal cord segmentation
  copy_gt_softseg "${file}" "${type}"

  # Copy GT disc labels segmentation
  copy_gt_disc_labels "${file}" "${type}"

  # Label vertebral levels in the native resolution
  label_vertebrae ${file} 't2'

  # rename contrasts 
  if [[ $contrast == "flip-1_mt-on_MTS" ]]; then
    contrast="MTon"
    deepseg_input_c="t2s"
  elif [[ $contrast == "flip-2_mt-off_MTS" ]]; then
    contrast="MToff"
    deepseg_input_c="t1"
  elif [[ $contrast == "rec-average_dwi" ]]; then
    contrast="DWI"
    deepseg_input_c="dwi"
  elif [[ $contrast == "T1w" ]]; then
    deepseg_input_c="t1"
  elif [[ $contrast == "T2w" ]]; then
    deepseg_input_c="t2"
  elif [[ $contrast == "T2star" ]]; then
    deepseg_input_c="t2s"
  fi

  # # 1. Compute (soft) CSA of the original soft GT
  # # renaming file so that it can be fetched from the CSA csa file later 
  # FILEINPUT="${file%%_*}_${contrast}_softseg_soft"
  # cp ${file}_softseg.nii.gz ${FILEINPUT}.nii.gz
  # sct_process_segmentation -i ${FILEINPUT}.nii.gz -vert 2:4 -vertfile ${file}_seg-manual_labeled.nii.gz -o $PATH_RESULTS/csa_label_types_c24.csv -append 1

  # Threshold the soft GT 
  FILETHRESH="${file%%_*}_${contrast}_softseg_bin"
  sct_maths -i ${file}_softseg.nii.gz -bin 0.5 -o ${FILETHRESH}.nii.gz

  # 2. Compute CSA of the binarized soft GT 
  sct_process_segmentation -i ${FILETHRESH}.nii.gz -vert 2:3 -vertfile ${file}_seg-manual_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}.csv -append 1

  # 3. Segment SC using different methods, binarize at 0.5 and compute CSA
	CUDA_VISIBLE_DEVICES=1 segment_sc_MONAI ${file} "${file}_seg-manual" 'monai' ${contrast} ${csv_fname}
  CUDA_VISIBLE_DEVICES=1 segment_sc_MONAI ${file} "${file}_seg-manual" 'mednext' ${contrast} ${csv_fname}
  CUDA_VISIBLE_DEVICES=3 segment_sc_MONAI ${file} "${file}_seg-manual" 'unetr' ${contrast} ${csv_fname}
  CUDA_VISIBLE_DEVICES=3 segment_sc_MONAI ${file} "${file}_seg-manual" 'swinunetr' ${contrast} ${csv_fname}
  CUDA_VISIBLE_DEVICES=1 segment_sc_MONAI ${file} "${file}_seg-manual" 'swinpretrained' ${contrast} ${csv_fname}
  CUDA_VISIBLE_DEVICES=3 segment_sc_nnUNet ${file} "${file}_seg-manual" '3d_fullres' ${contrast} ${csv_fname}
  segment_sc ${file} "${file}_seg-manual" 'deepseg' ${deepseg_input_c} ${contrast} ${csv_fname}
  # TODO: run on deep/progseg after fixing the contrasts for those
  # segment_sc ${file_res} 't2' 'propseg' '' "${file}_seg-manual" ${native_res}

  # # 3.1 Ensemble the predictions from different models
  # segment_sc_ensemble ${file} "${file}_seg-manual" ${contrast} ${csv_fname}

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