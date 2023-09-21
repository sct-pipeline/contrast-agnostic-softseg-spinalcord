#!/bin/bash
#
# Compare sct_propseg and sct_deepseg_sc
# Note: subjects from both datasets have to be located in the same BIDS-like folder, example:
# ├── derivatives
# │	 └── labels
# │	     ├── sub-5416   # sci-colorado subject
# │	     │	 └── anat
# │	     │	     ├── sub-5416_T2w_lesion-manual.json
# │	     │	     ├── sub-5416_T2w_lesion-manual.nii.gz
# │	     │	     ├── sub-5416_T2w_seg-manual.json
# │	     │	     └── sub-5416_T2w_seg-manual.nii.gz
# │	     └── sub-zh01   # sci-zurich subject
# │	         └── ses-01
# │	             └── anat
# │	                 ├── sub-zh01_ses-01_acq-sag_T2w_lesion-manual.json
# │	                 ├── sub-zh01_ses-01_acq-sag_T2w_lesion-manual.nii.gz
# │	                 ├── sub-zh01_ses-01_acq-sag_T2w_seg-manual.json
# │	                 └── sub-zh01_ses-01_acq-sag_T2w_seg-manual.nii.gz
# ├── sub-5416    # sci-colorado subject
# │	 └── anat
# │	     ├── sub-5416_T2w.json
# │	     └── sub-5416_T2w.nii.gz
# └── sub-zh01    # sci-zurich subject
#    └── ses-01
#        └── anat
#            ├── sub-zh01_ses-01_acq-sag_T2w.json
#            └── sub-zh01_ses-01_acq-sag_T2w.nii.gz
#
# Note: conda environment with nnUNetV2 is required to run this script.
# For details how to install nnUNetV2, see:
# https://github.com/ivadomed/utilities/blob/main/quick_start_guides/nnU-Net_quick_start_guide.md#installation
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
#  "script_args" : "<PATH_TO_REPO>/model_seg_sci/packaging/run_inference_single_subject.py <PATH_TO_MODEL>/sci-multisite-model"
# }
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Author: Jan Valosek
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
#PATH_NNUNET_SCRIPT=$2
#PATH_NNUNET_MODEL=$3

echo "SUBJECT: ${SUBJECT}"
#echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
#echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------
# Get ANIMA binaries path
#anima_binaries_path=$(grep "^anima = " ~/.anima/config.txt | sed "s/.* = //" | sed 's/\/$//')
anima_binaries_path=$HOME/.anima/
# Check if manual label already exists. If it does, copy it locally.
# NOTE: manual disc labels should go from C1-C2 to C7-T1.
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  local ofolder="$3"
  # Update global variable with segmentation file name
  FILELABEL="${file}_discs"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${FILELABEL}.nii.gz"
  
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c t2 -ofolder $ofolder
  else
    echo "Not found. cannot compute CSA."
  fi
}



# Segment spinal cord and compute ANIMA segmentation performance metrics
segment_sc() {
  local file="$1"
  local contrast="$2"
  local method="$3"     # deepseg or propseg
  local kernel="$4"     # 2d or 3d; only relevant for deepseg
  local PATH_QC_SEG="$5"
  local type="$6"
  # Segment spinal cord
  if [[ $method == 'deepseg' ]];then
      FILESEG="${file}_seg_${method}_${kernel}"

      # Get the start time
      start_time=$(date +%s)
      # Run SC segmentation
      sct_deepseg_sc -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -kernel ${kernel} -qc ${PATH_QC_SEG} -qc-subject ${SUBJECT}
      # Get the end time
      end_time=$(date +%s)
      # Calculate the time difference
      execution_time=$(python3 -c "print($end_time - $start_time)")
      echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

      # Compute ANIMA segmentation performance metrics
      compute_anima_metrics ${FILESEG} ${file}_seg.nii.gz

  elif [[ $method == 'propseg' ]]; then
      FILESEG="${file}_seg_${method}"

      # Get the start time
      start_time=$(date +%s)
      # Run SC segmentation
      cd $type
      sct_propseg -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -qc ${PATH_QC_SEG} -qc-subject ${SUBJECT}
      # Get the end time
      end_time=$(date +%s)
      # Calculate the time difference
      execution_time=$(python3 -c "print($end_time - $start_time)")
      echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

      # Remove centerline (we don't need it)
      rm ${file}_centerline.nii.gz
      cd ..
      # Compute ANIMA segmentation performance metrics
      compute_anima_metrics ${FILESEG} ${file}_seg.nii.gz
  fi
}

# Compute ANIMA segmentation performance metrics
compute_anima_metrics(){
  # We have to copy qform matrix from seg-manual to the automatically generated segmentation to avoid ITK error:
  # "Description: ITK ERROR: SegmentationMeasuresImageFilter(): Inputs do not occupy the same physical space!"
  # Related to the following issue : https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4135
  sct_image -i ${file}_seg.nii.gz -copy-header ${FILESEG}.nii.gz -o ${FILESEG}_updated_header.nii.gz

  # Compute ANIMA segmentation performance metrics
  # -i : input segmentation
  # -r : GT segmentation
  # -o : output file
  # -d : surface distances evaluation
  # -s : compute metrics to evaluate a segmentation
  # -X : stores results into a xml file.
  ${anima_binaries_path}/animaSegPerfAnalyzer -i ${FILESEG}_updated_header.nii.gz -r ${file}_softseg_bin.nii.gz -o ${PATH_RESULTS}/${FILESEG} -d -s -X
  rm ${FILESEG}_updated_header.nii.gz
}

# Copy GT segmentation
copy_gt(){
  local file="$1"
  # Construct file name to GT segmentation located under derivatives/labels
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_seg-manual.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILESEGMANUAL ${file}_seg-manual.nii.gz
  else
      echo "File ${FILESEGMANUAL}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual GT segmentation ${FILESEGMANUAL}.nii.gz does not exist. Exiting."
      exit 1
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

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# get starting time:
start=`date +%s`

# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Create directory
mkdir -p $PATH_DATA_PROCESSED/$SUBJECT

# Go to subject folder for source images
cd ${SUBJECT}


# Initialize filenames
file_t1="${SUBJECT}_T1w"
file_t2="${SUBJECT}_T2w"
file_t2s="${SUBJECT}_T2star"
file_t1w="${SUBJECT}_flip-2_mt-off_MTS"
file_mton="${SUBJECT}_flip-1_mt-on_MTS"
file_dwi_mean="${SUBJECT}_rec-average_dwi"
contrasts=($file_t1 $file_t2 $file_t2s $file_t1w $file_mton $file_dwi_mean)

for file_path in "${contrasts[@]}";do
  # Find contrast to do compute CSA
  if [[ $file_path == *"flip-2_mt-off_MTS"* ]];then
    contrast_seg="flip-2_mt-off_MTS"
    contrast="t1"
  elif [[ $file_path == *"T2star"* ]];then
    contrast_seg="T2star"
    contrast="t2s"
  elif [[ $file_path == *"T2w"* ]];then
    contrast_seg="T2w"
    contrast="t2"
  elif [[ $file_path == *"T1w"* ]];then
    contrast_seg="T1w"
    contrast="t1"  # For segmentation
  elif [[ $file_path == *"flip-1_mt-on_MTS"* ]];then
    contrast_seg="flip-1_mt-on_MTS"
    contrast="t2s"
  elif [[ $file_path == *"dwi"* ]];then
    contrast_seg="dwi"
    contrast="dwi"
  fi

  # Find if anat or dwi folder
  type=$(find_contrast $file_path)
  file=${file_path/#"$type"}  # add sub folder in file name
  file_path=${type}$file
  mkdir -p $PATH_DATA_PROCESSED/$SUBJECT/${type}
  # Copy source images
  # Note: we use '/./' in order to include the sub-folder 'ses-0X'
  rsync -Ravzh ${PATH_DATA}/${SUBJECT}/${file_path}.* .

  # Get manual hard GT to get labeled segmentation
  FILESEG="${file_path}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL "${FILESEG}.nii.gz"
  fi

  # Get manual SOFT GT to get labeled segmentation
  FILESEGSOFT="${file_path}_softseg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels_softseg/${SUBJECT}/${FILESEGSOFT}.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL "${FILESEGSOFT}.nii.gz"
  fi
  # Binarize softseg
  sct_maths -i ${FILESEGSOFT}.nii.gz -bin 0.5 -o ${FILESEGSOFT}_bin.nii.gz
  
  # Create labeled segmentation of vertebral levels (only if it does not exist) 
  label_if_does_not_exist $file_path $FILESEG $type

  file_seg_labeled="${FILESEG}_labeled"

  # Segment SC using different methods and compute ANIMA segmentation performance metrics
  # deepseg_sc 2D
  ###################
  mkdir -p $PATH_QC/deepseg2d
  segment_sc "${file_path}" $contrast 'deepseg' '2d' $PATH_QC/deepseg2d
  pred_seg=${file_path}_seg_deepseg_2d
  # Compute CSA
  # Compute average cord CSA between C2 and C3
  mkdir -p ${PATH_RESULTS}/deepseg2d
  sct_process_segmentation -i ${pred_seg}.nii.gz -vert 2:3 -vertfile ${file_seg_labeled}.nii.gz -o ${PATH_RESULTS}/deepseg2d/csa_pred_${contrast_seg}.csv -append 1

  # deepseg_sc 3D
  ###################
  mkdir -p $PATH_QC/deepseg3d
  segment_sc "${file_path}" $contrast 'deepseg' '3d' $PATH_QC/deepseg3d
  pred_seg=${file_path}_seg_deepseg_3d
  # Compute CSA
  # Compute average cord CSA between C2 and C3
  mkdir -p ${PATH_RESULTS}/deepseg3d
  sct_process_segmentation -i ${pred_seg}.nii.gz -vert 2:3 -vertfile ${file_seg_labeled}.nii.gz -o ${PATH_RESULTS}/deepseg3d/csa_pred_${contrast_seg}.csv -append 1
  
  # propseg
  ##################
  mkdir -p $PATH_QC/propseg
  segment_sc "${file}" $contrast 'propseg' '1' $PATH_QC/propseg "${type}"
  pred_seg=${file_path}_seg_propseg
  # Compute CSA
  # Compute average cord CSA between C2 and C3
  mkdir -p ${PATH_RESULTS}/propseg
  sct_process_segmentation -i ${pred_seg}.nii.gz -vert 2:3 -vertfile ${file_seg_labeled}.nii.gz -o ${PATH_RESULTS}/propseg/csa_pred_${contrast_seg}.csv -append 1

  # segment_sc_nnUNet "${file_t2}" '2d'
  # segment_sc_nnUNet "${file_t2}" '3d'
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
