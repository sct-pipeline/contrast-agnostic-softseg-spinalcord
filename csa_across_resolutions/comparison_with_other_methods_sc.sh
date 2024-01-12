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
PATH_MONAI_MODEL=$5     # path to the MONAI contrast-agnostic model

echo "SUBJECT: ${SUBJECT}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"
echo "PATH_MONAI_SCRIPT: ${PATH_MONAI_SCRIPT}"
echo "PATH_MONAI_MODEL: ${PATH_MONAI_MODEL}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------
# Get ANIMA binaries path
anima_binaries_path=$(grep "^anima = " ~/.anima/config.txt | sed "s/.* = //" | sed 's/\/$//')

# Check if manual label already exists. If it does, copy it locally.
# NOTE: manual disc labels should go from C1-C2 to C7-T1.
label_vertebrae(){
  local file="$1"
  local contrast="$2"
  # local file_seg="$2"
  # local ofolder="$3"
  
  # Update global variable with segmentation file name
  FILESEG="${file}_seg-manual"
  FILELABEL="${file}_discs"

  # Label vertebral levels
  # NOTE: the straightening step in sct_label_vertebrae is resulting in EmptyArray Error for sct_propseg
  # Hence, we're only using sct_label_utils to label vertebral levels instead
  sct_label_vertebrae -i ${file}.nii.gz -s ${FILESEG}.nii.gz -discfile ${FILELABEL}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  # sct_label_utils -i ${file}.nii.gz -disc ${FILELABEL}.nii.gz -o ${file_seg}_labeled.nii.gz

  # # Run QC
  # sct_qc -i ${file}.nii.gz -s ${file_seg}_labeled.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}
}


# Segment spinal cord using methods available in SCT (sct_deepseg_sc or sct_propseg)
segment_sc() {
  local file="$1"
  local contrast="$2"
  local method="$3"     # deepseg or propseg
  local kernel="$4"     # 2d or 3d; only relevant for deepseg

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

      # # Compute ANIMA segmentation performance metrics
      # compute_anima_metrics ${FILESEG} ${file}_seg-manual.nii.gz
  
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

      # # Compute ANIMA segmentation performance metrics
      # compute_anima_metrics ${FILESEG} ${file}_seg-manual.nii.gz

  fi

  # Create labeled segmentation of vertebral levels and compute CSA
  label_and_compute_csa $file $FILESEG
}


# Segment spinal cord using the contrast-agnostic nnUNet model
segment_sc_nnUNet(){
  local file="$1"
  local kernel="$2"     # 2d or 3d

  FILESEG="${file}_seg_nnunet_${kernel}"

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

  # # Compute ANIMA segmentation performance metrics
  # compute_anima_metrics ${FILESEG} ${file}_seg-manual.nii.gz

  # Create labeled segmentation of vertebral levels and compute CSA
  label_and_compute_csa $file $FILESEG

}

# Segment spinal cord using the MONAI contrast-agnostic model
segment_sc_MONAI(){
  local file="$1"

  FILESEG="${file}_seg_monai"

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MONAI_MODEL} --device gpu
  # Rename MONAI output
  mv ${file}_pred.nii.gz ${FILESEG}.nii.gz
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Binarize MONAI output (which is soft by default); output is overwritten
  sct_maths -i ${FILESEG}.nii.gz -bin 0.5 -o ${FILESEG}.nii.gz

  # Generate QC report
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  # # Compute ANIMA segmentation performance metrics
  # compute_anima_metrics ${FILESEG} ${file}_seg-manual.nii.gz

  # Create labeled segmentation of vertebral levels and compute CSA
  label_and_compute_csa $file $FILESEG
}

# # Compute ANIMA segmentation performance metrics
# compute_anima_metrics(){
#   # We have to copy qform matrix from seg-manual to the automatically generated segmentation to avoid ITK error:
#   # "Description: ITK ERROR: SegmentationMeasuresImageFilter(): Inputs do not occupy the same physical space!"
#   # Related to the following issue : https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4135
#   sct_image -i ${file}_seg-manual.nii.gz -copy-header ${FILESEG}.nii.gz -o ${FILESEG}_updated_header.nii.gz

#   # Compute ANIMA segmentation performance metrics
#   # -i : input segmentation
#   # -r : GT segmentation
#   # -o : output file
#   # -d : surface distances evaluation
#   # -s : compute metrics to evaluate a segmentation
#   # -X : stores results into a xml file.
#   ${anima_binaries_path}/animaSegPerfAnalyzer -i ${FILESEG}_updated_header.nii.gz -r ${file}_seg-manual.nii.gz -o ${PATH_RESULTS}/${FILESEG} -d -s -X

#   rm ${FILESEG}_updated_header.nii.gz
# }

# Copy GT spinal cord disc labels (located under derivatives/labels)
copy_gt_disc_labels(){
  local file="$1"
  # Construct file name to GT segmentation located under derivatives/labels
  FILEDISCLABELS="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_discs.nii.gz"
  echo ""
  echo "Looking for manual disc labels: $FILEDISCLABELS"
  if [[ -e $FILEDISCLABELS ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILEDISCLABELS ${file}_discs.nii.gz
  else
      echo "File ${FILEDISCLABELS}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual Disc Labels ${FILEDISCLABELS}.nii.gz does not exist. Exiting."
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

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# get starting time:
start=`date +%s`

# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source T2w images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*T2w.* .

# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# contrast
# ------------------------------------------------------------------------------
file="${SUBJECT}_T2w"

# Copy GT spinal cord segmentation
copy_gt_disc_labels "${file}"

# Check if file exists
if [[ ! -e ${file}.nii.gz ]]; then
    echo "File ${file}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file}.nii.gz does not exist. Exiting."
    exit 1
fi

# Get the native resolution of the image
native_res=$(sct_image -i ${file}.nii.gz -header | grep pixdim | awk -F'[][]' '{split($2, a, ", "); print a[2]"x"a[3]"x"a[4]}')

# Copy GT spinal cord segmentation
copy_gt_disc_labels "${file}"

# Copy GT spinal cord segmentation
copy_gt_seg "${file}"

# Label vertebral levels
label_vertebrae ${file} 't2'

# resolutions to be used for isotropic resampling
resolutions="1 1.25 1.5 1.75 2"

# Loop across resolutions
for res in ${resolutions}
do

  echo "Resampling image to ${res}mm isotropic resolution ..."
  # NOTE: the . in resolution is replaced by nothing to avoid issues with bash
  file_res="${file}_iso-${res/./}mm"
  # Resample image
  sct_resample -i ${file}.nii.gz -mm ${res}x${res}x${res} -x linear -o ${file_res}.nii.gz

  # NOTE that the resampled images do not have the disc labels. So, we (1) register the original image to the resampled image 
  # using -identity 1 to get the warping field, and (2) apply the warping field to the original disc labels to bring them to the 
  # space of the resampled image. The resampled disc labels are then used for vertebral labelling and CSA computation.

  # Register the original disc labels to the resampled image
  sct_register_multimodal -i ${file}.nii.gz -d ${file_res}.nii.gz -identity 1 -x linear

  # Apply the resulting warping field to the original disc labels
  sct_apply_transfo -i ${file}_discs.nii.gz -d ${file_res}.nii.gz -w warp_${file}2${file_res}.nii.gz -x label -o ${file_res}_discs.nii.gz

  # Segment SC using different methods and compute ANIMA segmentation performance metrics
  segment_sc_nnUNet ${file_res} '3d_fullres'
  segment_sc_MONAI ${file_res}
  segment_sc ${file_res} 't2' 'deepseg' '2d'
  # segment_sc "${file}_res-${res}mm" 't2' 'deepseg' '3d'
  segment_sc ${file_res} 't2' 'propseg'
  # segment_sc_nnUNet "${file}_res-${res}mm" '2d'
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
