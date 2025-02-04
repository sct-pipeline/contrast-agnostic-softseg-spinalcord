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
# PATH_NNUNET_SCRIPT=$2   # path to the nnUNet contrast-agnostic run_inference_single_subject.py
# PATH_NNUNET_MODEL=$3    # path to the nnUNet contrast-agnostic model
PATH_MONAI_SCRIPT=$2    # path to the MONAI contrast-agnostic run_inference_single_subject.py
PATH_NNUNET_SCRIPT=$3   # path to the nnUNet contrast-agnostic run_inference_single_subject.py
PATH_MONAI_MODELS=$4
# PATH_SWIN_MODEL=$5
PATH_SWDICE_SCRIPT="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/contrast-agnostic-softseg-spinalcord/monai/compute_slicewise_dice.py"

echo "SUBJECT: ${SUBJECT}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
# echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"
echo "PATH_MONAI_SCRIPT: ${PATH_MONAI_SCRIPT}"
# echo "PATH_MONAI_MODELS: ${PATH_MONAI_MODELS}"
# echo "PATH_SWIN_MODEL: ${PATH_SWIN_MODEL}"

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
  # sct_label_utils -i ${file}.nii.gz -disc ${FILELABEL}.nii.gz -o ${FILESEG}_labeled.nii.gz
  sct_label_utils -i ${FILESEG}.nii.gz -disc ${FILELABEL}.nii.gz -o ${FILESEG}_labeled.nii.gz

  # # Run QC
  # sct_qc -i ${file}.nii.gz -s ${file_seg}_labeled.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}
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

# Copy GT segmentation (located under derivatives/labels)
copy_gt_seg(){
  local file="$1"
  local type="$2"
  # Construct file name to GT segmentation located under derivatives/labels
  FILESEG="${PATH_DATA}/derivatives/labels/${SUBJECT}/${type}/${file}_label-SC_seg.nii.gz"
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
      FILEGT="${file%%_*}_${contrast_name}_softseg_bin"

      # Get the start time
      start_time=$(date +%s)
      # Run SC segmentation
      sct_deepseg_sc -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast_input} -kernel 2d -qc ${PATH_QC} -qc-subject ${SUBJECT}
      # Get the end time
      end_time=$(date +%s)
      # Calculate the time difference
      execution_time=$(python3 -c "print($end_time - $start_time)")
      echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv
      # Compute slicewise Dice
      slicewise_dice=$(python ${PATH_SWDICE_SCRIPT} --path-pred ${FILESEG}.nii.gz --path-gt ${FILEGT}.nii.gz)
      echo "${FILESEG},${slicewise_dice}" >> ${PATH_RESULTS}/slicewise_dice.csv

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

  # Compute CSA averaged across all slices C2-C3 vertebral levels for plotting the STD across contrasts
  sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:3 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}_c2c3.csv -append 1

  # Compute CSA "per slice" across _all_ slices in the SC for plotting the absolute CSA error across contrasts
  sct_process_segmentation -i ${FILESEG}.nii.gz -perslice 1 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}_${method}_perslice.csv -append 1

}

# Segment spinal cord using the contrast-agnostic nnUNet model
segment_sc_nnUNet(){
  local file="$1"
  local file_gt_vert_label="$2"
  local model="$3"     # 2d or 3d
  local contrast="$4"   # used only for saving output file name
  local csv_fname="$5"   # used for saving output file name
  local kernel="$6"    # 2d or 3d_fullres

  FILESEG="${file%%_*}_${contrast}_seg_${model}"
	PATH_MODEL=${PATH_MONAI_MODELS}/model_${model}
  # FILESEG="${file}_seg_nnunet${kernel}"
  # PATH_MODEL="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/sct_deployed_models/model_nnunet-AllRandInit${kernel}"

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILESEG}.nii.gz -path-model ${PATH_MODEL}/nnUNetTrainer__nnUNetPlans__${kernel} -pred-type sc -use-gpu -use-best-checkpoint
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Generate QC report
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Compute CSA averaged across all slices C2-C3 vertebral levels for plotting the STD across contrasts
  # NOTE: this is per-level because not all contrasts have thes same FoV (C2-C3 is what all contrasts have in common)
  sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:3 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}_c2c3.csv -append 1

  # Compute CSA "per slice" across _all_ slices in the SC for plotting the absolute CSA error across contrasts
  sct_process_segmentation -i ${FILESEG}.nii.gz -perslice 1 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}_${model}_perslice.csv -append 1

}

# Segment spinal cord using the MONAI contrast-agnostic model
segment_sc_MONAI(){
  local file="$1"
  local file_gt_vert_label="$2"
  local model="$3"     # monai or swinunetr or mednext
  local contrast="$4"   # used only for saving output file name
  local csv_fname="$5"   # used for saving output file name
  local models_384="v25 vPtrNewSeqLSciDcm vPtrVNSLSciDcm vPtrVNSLSciDcm_ColZurLes vPtrVNSLSciDcm_All"

	if [[ " ${models_384[@]} " =~ " ${model} " ]]; then
    FILESEG="${file%%_*}_${contrast}_seg_${model}"
		PATH_MODEL=${PATH_MONAI_MODELS}/model_${model}
    model_name='monai'
    max_feat=384
    pad='edge'
  else
    FILESEG="${file%%_*}_${contrast}_seg_${model}"
		PATH_MODEL=${PATH_MONAI_MODELS}/model_${model}
    model_name='monai'
    max_feat=320
    pad='edge'

	fi

  FILEGT="${file%%_*}_${contrast}_softseg_bin"

  # Get the start time
  start_time=$(date +%s)
  echo "Running inference from model at ${PATH_MODEL}"
  # NOTE: surprisingly, the `edge` padding is resulting in higher abs. csa error compared to `constant` (zero) padded inputs.
  # Run SC segmentation
  python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MODEL} --device gpu --model ${model_name} --pred-type soft --pad-mode ${pad} --max-feat ${max_feat}
  # python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MODEL} --device gpu --model monai --pred-type soft --pad-mode constant
  # Rename MONAI output
  mv ${file}_pred.nii.gz ${FILESEG}.nii.gz
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Binarize MONAI output (which is soft by default); output is overwritten
  sct_maths -i ${FILESEG}.nii.gz -bin 0.5 -o ${FILESEG}.nii.gz

  # Compute slicewise Dice
  slicewise_dice=$(python ${PATH_SWDICE_SCRIPT} --path-pred ${FILESEG}.nii.gz --path-gt ${FILEGT}.nii.gz)
  echo "${FILESEG},${slicewise_dice}"
  echo "${FILESEG},${slicewise_dice}" >> ${PATH_RESULTS}/slicewise_dice.csv

  # Generate QC report 
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Compute CSA averaged across all slices C2-C3 vertebral levels for plotting the STD across contrasts
  # NOTE: this is per-level because not all contrasts have thes same FoV (C2-C3 is what all contrasts have in common)
  sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:3 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}_c2c3.csv -append 1

  # Compute CSA "per slice" across _all_ slices in the SC for plotting the absolute CSA error across contrasts
  sct_process_segmentation -i ${FILESEG}.nii.gz -perslice 1 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}_${model}_perslice.csv -append 1

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
  sct_maths -i ${FILETEMP}_seg_monai_orig.nii.gz -add ${FILETEMP}_seg_monai_ll.nii.gz -o ${FILESEG}.nii.gz
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
rsync -Ravzh ${PATH_DATA}/derivatives/data_preprocessed/./${SUBJECT}/anat/* .
# copy DWI data
rsync -Ravzh ${PATH_DATA}/derivatives/data_preprocessed/./${SUBJECT}/dwi/* .

# ------------------------------------------------------------------------------
# contrast
# ------------------------------------------------------------------------------
contrasts="space-other_T1w space-other_T2w space-other_T2star flip-1_mt-on_space-other_MTS flip-2_mt-off_space-other_MTS rec-average_dwi"
# contrasts="space-other_T1w rec-average_dwi"

# output csv filename
csv_fname="csa_deploy_vs_unbalanced"   # "csa_label_inputs"

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
  if [[ $contrast == "flip-1_mt-on_space-other_MTS" ]]; then
    contrast="MTon"
    deepseg_input_c="t2s"
  elif [[ $contrast == "flip-2_mt-off_space-other_MTS" ]]; then
    contrast="MToff"
    deepseg_input_c="t1"
  elif [[ $contrast == "rec-average_dwi" ]]; then
    contrast="DWI"
    deepseg_input_c="dwi"
  elif [[ $contrast == "space-other_T1w" ]]; then
    contrast="T1w"
    deepseg_input_c="t1"
  elif [[ $contrast == "space-other_T2w" ]]; then
    contrast="T2w"
    deepseg_input_c="t2"
  elif [[ $contrast == "space-other_T2star" ]]; then
    contrast="T2star"
    deepseg_input_c="t2s"
  fi

  # Copy GT spinal cord segmentation
  copy_gt_seg "${file}" "${type}"

  # Copy soft GT spinal cord segmentation
  copy_gt_softseg_bin "${file}" "${type}"

  # Copy GT disc labels segmentation
  copy_gt_disc_labels "${file}" "${type}" "${contrast}"

  # Label vertebral levels in the native resolution
  label_vertebrae ${file} 't2'

  # Rename the softseg_bin GT with the shorter contrast name
  FILEBIN="${file%%_*}_${contrast}_softseg_bin"
  mv ${file}_softseg_bin.nii.gz ${FILEBIN}.nii.gz
  # sct_maths -i ${file}_softseg.nii.gz -bin 0.5 -o ${FILETHRESH}.nii.gz

  # Generate QC report 
  sct_qc -i ${file}.nii.gz -s ${FILEBIN}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # 2. Compute CSA of the binarized soft GT
  # 2.1 Compute CSA averaged across all slices C2-C3 vertebral levels for plotting the STD across contrasts
  # NOTE: this is per-level because not all contrasts have thes same FoV (C2-C3 is what all contrasts have in common)
  sct_process_segmentation -i ${FILEBIN}.nii.gz -vert 2:3 -vertfile ${file}_seg-manual_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}_c2c3.csv -append 1

  # 2.2 Compute CSA "per slice" across _all_ slices in the SC for plotting the absolute CSA error across contrasts
  sct_process_segmentation -i ${FILEBIN}.nii.gz -perslice 1 -vertfile ${file}_seg-manual_labeled.nii.gz -o $PATH_RESULTS/${csv_fname}_softseg_bin_perslice.csv -append 1

  # 3. Segment SC using different methods, binarize at 0.5 and compute CSA
	# CUDA_VISIBLE_DEVICES=0 segment_sc_MONAI ${file} "${file}_seg-manual" 'v20' ${contrast} ${csv_fname}
  # CUDA_VISIBLE_DEVICES=0 segment_sc_MONAI ${file} "${file}_seg-manual" 'v25' ${contrast} ${csv_fname}
  CUDA_VISIBLE_DEVICES=3 segment_sc_nnUNet ${file} "${file}_seg-manual" 'nnunet-AllRandInit2D' ${contrast} ${csv_fname} '2d'
  CUDA_VISIBLE_DEVICES=2 segment_sc_nnUNet ${file} "${file}_seg-manual" 'nnunet-AllRandInit3D' ${contrast} ${csv_fname} '3d_fullres'
  CUDA_VISIBLE_DEVICES=0 segment_sc_MONAI ${file} "${file}_seg-manual" 'vPtrV21-allWithPraxWithSCT' ${contrast} ${csv_fname}
  # CUDA_VISIBLE_DEVICES=3 segment_sc_nnUNet ${file} "${file}_seg-manual" '3d' ${contrast} ${csv_fname}
  # segment_sc ${file} "${file}_seg-manual" 'deepseg' ${deepseg_input_c} ${contrast} ${csv_fname}

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
