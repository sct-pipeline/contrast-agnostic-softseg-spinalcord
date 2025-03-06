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
# PATH_NNUNET_SCRIPT=$3   # path to the nnUNet contrast-agnostic run_inference_single_subject.py
# PATH_NNUNET_MODEL=$4    # path to the nnUNet contrast-agnostic model
PATH_MONAI_SCRIPT=$3    # path to the MONAI contrast-agnostic run_inference_single_subject.py
PATH_NNUNET_SCRIPT=$4
# PATH_MONAI_MODEL=$4     

echo "SUBJECT: ${SUBJECT}"
echo "QC_DATASET: ${QC_DATASET}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
# echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"
echo "PATH_MONAI_SCRIPT: ${PATH_MONAI_SCRIPT}"
echo "PATH_MONAI_MODEL: ${PATH_MONAI_MODEL}"

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
      # check if json file exists
      if [[ -e ${FILESEG/.nii.gz/.json} ]]; then
          rsync -avzh ${FILESEG/.nii.gz/.json} ${file}_seg-manual.json
      else
          echo "${FILESEG/.nii.gz/.json} does not exist" >> ${PATH_LOG}/missing_files.log
      fi
  else
      echo "File ${FILESEG}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual Segmentation ${FILESEG} does not exist. Exiting."
      # exit 1
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
      sct_deepseg_sc -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -kernel ${kernel} # -qc ${PATH_QC} -qc-subject ${SUBJECT}
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

  elif [[ $method == 'SCIsegV2' ]];then
      FILESEG="${file}_seg_${method}"

      # Get the start time
      start_time=$(date +%s)
      # Run SC segmentation
      SCT_USE_GPU=1 sct_deepseg -task seg_sc_lesion_t2w_sci -i ${file}.nii.gz -o ${FILESEG}.nii.gz 
      # rename output
      mv ${FILESEG}_sc_seg.nii.gz ${FILESEG}.nii.gz
      # Get the end time
      end_time=$(date +%s)
      # Calculate the time difference
      execution_time=$(python3 -c "print($end_time - $start_time)")
      echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  fi

  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -d ${FILESEG}.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject ${SUBJECT}
  # # Compute CSA from the the SC segmentation resampled back to native resolution using the GT vertebral labels
  # sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:4 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/csa_label_types_c24.csv -append 1

}

# Segment spinal cord using the contrast-agnostic nnUNet model
segment_sc_nnUNet(){
  local file="$1"
  local model="$2"     # monai, swinunetr, mednext
  local kernel="$3"     # 2d or 3d

  if [[ $kernel == '2D' ]]; then
    kernel_fmt='2d'
  elif [[ $kernel == '3D' ]]; then
    kernel_fmt='3d_fullres'
  fi

  # FILESEG="${file}_seg_nnunet_${kernel}"
  FILESEG="${file}_seg_${model}${kernel}"
  PATH_MODEL="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/sct_deployed_models/model_${model}${kernel}"

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILESEG}.nii.gz -path-model ${PATH_MODEL}/nnUNetTrainer__nnUNetPlans__${kernel_fmt} -pred-type sc -use-gpu -use-best-checkpoint
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Generate QC report
  # sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -d ${FILESEG}.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # # Compute CSA from the prediction resampled back to native resolution using the GT vertebral labels
  # sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:4 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/csa_label_types_c24.csv -append 1

}

# Segment spinal cord using the MONAI contrast-agnostic model
segment_sc_MONAI(){
  local file="$1"
  # local label_type="$2"     # soft or soft_bin
  local model="$2"     # monai, swinunetr, mednext
  local models_384="v25 vPtrVNSLSciDcm_ColZurLes vPtrNewSeqLSciDcm"
  local models_320="v20 v21 vPtrV21-allNoPraxNoSCT vPtrV21-allNoPraxWithSCT vPtrV21-allWithPraxNoSCT vPtrV21-allWithPraxWithSCT"

	# if [[ $label_type == 'soft' ]]; then
	# 	FILEPRED="${file}_seg_monai_soft_input"
	# 	PATH_MODEL=${PATH_MONAI_MODEL_SOFT}
	
	# elif [[ $label_type == 'bin' ]]; then
  #   FILEPRED="${file}_seg_monai_bin_input"
	# 	PATH_MODEL=${PATH_MONAI_MODEL_BIN}
	
	# fi
	if [[ " ${models_384[@]} " =~ " ${model} " ]]; then
		# FILEPRED="${file}_seg_${model}"
    # PATH_MODEL=${PATH_MONAI_MODEL}
    if [[ $model == 'v25' ]]; then
      PATH_MODEL="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/sct_deployed_models/model_v25"
      FILEPRED="${file}_seg_${model}"
    elif [[ $model == 'vPtrNewSeqLSciDcm' ]]; then
      PATH_MODEL="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/sct_deployed_models/model_vPtrNewSeqLSciDcm"
    elif [[ $model == 'vPtrVNSLSciDcm_ColZurLes' ]]; then
      PATH_MODEL="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/sct_deployed_models/model_vPtrVNSLSciDcm_ColZurLes"
      FILEPRED="${file}_seg_M2"
    fi
    echo "Model: ${model};  Using model checkpoint in: ${PATH_MODEL}"
    max_feat=384
  
  elif [[ " ${models_320[@]} " =~ " ${model} " ]]; then
    # FILEPRED="${file}_seg_${model}"
    # PATH_MODEL=${PATH_MONAI_MODEL}
    PATH_MODEL="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/sct_deployed_models/model_${model}"
    if [[ $model == 'v21' ]]; then
      FILEPRED="${file}_seg_${model}"
    elif [[ $model == 'vPtrV21-allNoPraxNoSCT' ]]; then
      FILEPRED="${file}_seg_M2prime"
    elif [[ $model == 'vPtrV21-allNoPraxWithSCT' ]]; then
      FILEPRED="${file}_seg_M3prime"
    elif [[ $model == 'vPtrV21-allWithPraxNoSCT' ]]; then
      FILEPRED="${file}_seg_M4prime"
    elif [[ $model == 'vPtrV21-allWithPraxWithSCT' ]]; then
      FILEPRED="${file}_seg_M5prime"
    fi
    max_feat=320
	
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
  # sct_qc -i ${file}.nii.gz -s ${FILEPRED}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  sct_qc -i ${file}.nii.gz -s ${FILEPRED}.nii.gz -d ${FILEPRED}.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject ${SUBJECT}
  # # compute ANIMA metrics
  # compute_anima_metrics ${FILEPRED} ${file}_seg-manual

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

praxis_datasets="site_009 site_012 site_013 site_014 site_03"

# Note: we use '/./' in order to include the sub-folder 'ses-0X'
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
if [[ $QC_DATASET == "sci-colorado" ]]; then
  contrasts="T2w"
  label_suffix="seg-manual"
  deepseg_input_c="t2"

elif [[ $QC_DATASET == "basel-mp2rage" ]]; then
  contrasts="UNIT1"
  label_suffix="label-SC_seg"
  deepseg_input_c="t1"

elif [[ $QC_DATASET == "dcm-zurich" ]]; then
  contrasts="acq-axial_T2w"
  label_suffix="label-SC_mask-manual"
  deepseg_input_c="t2"

elif [[ $QC_DATASET == "ms-basel-2020" ]]; then
  contrasts="acq-sagCervTSE_PD"
  label_suffix="seg-manual"
  deepseg_input_c="t2"

elif [[ $QC_DATASET == "canproco" ]]; then
  contrasts="PSIR STIR"
  label_suffix="seg-manual"
  deepseg_input_c="t2"

elif [[ $QC_DATASET == "sct-testing-large" ]]; then
  contrasts="T2star acq-MTon_MTR acq-dwiMean_dwi"
  label_suffix="seg-manual"
  deepseg_input_c="t2"

elif [[ $QC_DATASET == "whole-spine" ]]; then
  contrasts="T1w"
  label_suffix="label-SC_seg"
  deepseg_input_c="t1"

elif [[ $QC_DATASET == "nih-ms-mp2rage" ]]; then
  contrasts="UNIT1 T1map"
  label_suffix="label-SC_seg" 

elif [[ " ${praxis_datasets[@]} " =~ " ${QC_DATASET} " ]]; then
  contrasts="acq-STIRsag_T2w acq-sag_T2w acq-sag_run-01_T2w"
  label_suffix="label-SC_seg" 
  deepseg_input_c="t2"

elif [[ $QC_DATASET == "data-single-subject-from-duke" ]]; then
  contrasts="rec-average_dwi"
  label_suffix="label-SC_seg"

elif [[ $QC_DATASET == "difficult-cases" ]]; then
  contrasts="T1w T2w T2star flip-1_mt-on_MTS flip-2_mt-off_MTS rec-average_dwi anatGRE UNIT1 PSIR STIR"
  label_suffix="label-SC_seg"

elif [[ $QC_DATASET == "sci-zurich" ]]; then
  contrasts="acq-sag_T2w"
  label_suffix="label-SC_seg"
  deepseg_input_c="t2"

else
  echo "ERROR: Dataset ${QC_DATASET} not recognized. Exiting."
  exit 1

fi


# Loop across contrasts
for contrast in ${contrasts}; do
    
  echo "Contrast: ${contrast}"

  if [[ $contrast == "rec-average_dwi" ]]; then
    fldr="dwi"
  else
    fldr="anat"
  fi

  PATH_DERIVATIVES="${PATH_DATA}/derivatives/labels/./${SUBJECT}/${fldr}"
  PATH_IMAGES="${PATH_DATA}/./${SUBJECT}/${fldr}"

  # # find all the files ending with the contrast name (IF GT labels exist)
  # mapfile -t files < <(find ${PATH_DERIVATIVES} -name "*${contrast}_${label_suffix}.nii.gz")
  
  # find all the files ending with the contrast name (IF GT labels DO NOT exist)
  mapfile -t files < <(find ${PATH_IMAGES} -name "*${contrast}.nii.gz")

  for label_file in "${files[@]}"; do
    # Process each file
    echo "$label_file"

    # NOTE: this replacement is cool because it automatically takes care of 'ses-XX' for longitudinal data
    file="${SUBJECT//[\/]/_}_*${contrast}"

    # check if label exists in the dataset
    if [[ ! -f ${label_file} ]]; then
        echo "Label File ${label_file} does not exist" >> ${PATH_LOG}/missing_files.log
        continue

    else
        echo "Label File ${label_file} exists. Proceeding..."
        
        # # copy labels
        # rsync -Ravzh ${PATH_DERIVATIVES}/${file}_${label_suffix}.nii.gz .
        # file=$(basename ${label_file} | sed "s/_${label_suffix}.nii.gz//")
        file=$(basename ${label_file} | sed "s/.nii.gz//")    # (IF copying only images)
        # copy source images
        rsync -Ravzh ${PATH_IMAGES}/${file}.nii.gz .
    fi
    
    cd ${PATH_DATA_PROCESSED}/${SUBJECT}/${fldr}

    # # Copy GT spinal cord segmentation
    # copy_gt_seg "${file}" "${label_suffix}"

    # # Generate QC report the GT spinal cord segmentation
    # # sct_qc -i ${file}.nii.gz -s ${file}_seg-manual.nii.gz -d ${file}_seg-manual.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject ${SUBJECT}
    # sct_qc -i ${file}.nii.gz -s ${file}_seg-manual.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

    # Segment SC using different methods, binarize at 0.5 and compute QC
    # CUDA_VISIBLE_DEVICES=1 segment_sc ${file} 'SCIsegV2'
    # CUDA_VISIBLE_DEVICES=1 segment_sc_MONAI ${file} 'v20'
    # CUDA_VISIBLE_DEVICES=2 segment_sc_nnUNet ${file} 'nnunet-AllRandInit' '2D'
    CUDA_VISIBLE_DEVICES=3 segment_sc_nnUNet ${file} 'nnunet-AllRandInit' '3D'
    # CUDA_VISIBLE_DEVICES=2 segment_sc_nnUNet ${file} 'nnunet-AllInferred' '3D'
    # CUDA_VISIBLE_DEVICES=1 segment_sc_MONAI ${file} 'v25'
    # CUDA_VISIBLE_DEVICES=2 segment_sc_MONAI ${file} 'vPtrV21-allWithPraxWithSCT'           # model M5prime
    segment_sc ${file} 'deepseg' ${deepseg_input_c}

#     # Create new "_clean" folder with BIDS-updated derivatives filenames
#     date_time=$(date +"%Y-%m-%d %H:%M:%S")
#     json_dict='{
#       "GeneratedBy": [
#         {
#           "Name": "contrast-agnostic-softseg-spinalcord",
#           "Version": "vPtrV21noSCT",
#           "Date": "'$date_time'"
#         }
#       ]
#     }'

#     PATH_DATA_PROCESSED_CLEAN="${PATH_DATA_PROCESSED}_clean"
#     # create new folder and copy only the predictions
#     mkdir -p ${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat

#     # rsync -avzh ${file}_seg_v25.nii.gz ${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat/${file%%_*}_space-other_${contrast}_desc-softseg_label-SC_seg.nii.gz
#     # rsync -avzh ${file}_seg_v25.nii.gz ${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat/${file%%_*}_${contrast}_desc-softseg_label-SC_seg.nii.gz
#     rsync -avzh ${file}_seg_M2prime.nii.gz ${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat/${file}_desc-softseg_label-SC_seg.nii.gz
#     rsync -avzh ${file}_seg-manual.json ${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat/${file}_desc-softseg_label-SC_seg.json

#     # create json file
#     # echo $json_dict > ${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat/${file%%_*}_${contrast}_desc-softseg_label-SC_seg.json
#     echo $json_dict > ${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat/${file}_desc-softseg_label-SC_seg.json
#     # re-save json files with indentation
#     python -c "import json; 
# json_file = '${PATH_DATA_PROCESSED_CLEAN}/derivatives/labels_softseg_bin/${SUBJECT}/anat/${file}_desc-softseg_label-SC_seg.json';
# with open(json_file, 'r') as f: 
#   data = json.load(f);
#   json.dump(data, open(json_file, 'w'), indent=4);
# "

  cd ${PATH_DATA_PROCESSED}
  
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
