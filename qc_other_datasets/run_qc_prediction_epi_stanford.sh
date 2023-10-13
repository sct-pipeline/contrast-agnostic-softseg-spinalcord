#!/bin/bash
#
# Run QC report on prediction masks
# Usage:
#   ./run_qc_prediction_epi_stanford.sh <SUBJECT>
#
#
# Authors: Sandrine Bédard (Adapted for EPI data by Naga Karthik)
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

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

# Retrieve input params
SUBJECT=$1
PATH_PRED_SEG=$2
MODEL=$3

echo "PATH_PRED_SEG: ${PATH_PRED_SEG}"
echo "MODEL: ${MODEL}"

# get starting time:
start=`date +%s`

# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd ${PATH_DATA_PROCESSED}
# Copy list of participants in processed data folder
if [[ ! -f "participants.tsv" ]]; then
  # NOTE: EPI data are mean images and not the original image, hence they are stored under derivatives
  # (PATH_DATA points to derivatives)
  rsync -avzh $PATH_DATA/../../participants.tsv .
fi
# Copy list of participants in resutls folder
if [[ ! -f $PATH_RESULTS/"participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/../../participants.tsv $PATH_RESULTS/"participants.tsv"
fi

# Copy source images
# Copy session if specified
if [[ $SUBJECT == *"ses"* ]];then
    rsync -Ravzh $PATH_DATA/./$SUBJECT .
else
    rsync -avzh $PATH_DATA/$SUBJECT .
fi

# SCRIPT STARTS HERE
# ==============================================================================
# Go to anat folder where all structural data are located
cd ${SUBJECT}
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_sub="${SUBJECT//[\/]/_}"

# if model is nnUNet, then we run the following code
if [[ ${MODEL} == "nnUNet" ]]; then
  echo "Running QC for nnUNet predictions ..."

  for file_pred in ${PATH_PRED_SEG}/*.nii.gz; do
    echo $file_pred
    if [[ $file_pred == *$file_sub* ]];then
      echo " File found, running QC report $file_pred"
      # Find if anat or dwi
      file_seg_basename=${file_pred##*/}
      echo $file_seg_basename
      # type=$(find_contrast $file_pred)
      type=func
      prefix="stanfordEPI_"  # TODO change accroding to prediction names
      file_image=${file_seg_basename#"$prefix"}
      file_image="${file_image::-11}"  # Remove X.nii.gz since the number X varies
      # split with "-"
      arrIN=(${file_image//-/ })
      contrast="_task-rest_desc-mocomean_bold"
      file_image=${arrIN[0]}"-"${arrIN[1]}"${contrast}.nii.gz"
      echo $file_image
      # rsync prediction mask
      file_pred_new_name=${type}/${arrIN[0]}"-"${arrIN[1]}"${contrast}_pred.nii.gz"
      # file_pred_new_name=${type}/${file_image}_pred.nii.gz
      rsync -avzh $file_pred $file_pred_new_name
      # Create QC for pred mask
      sct_qc -i ${type}/${file_image} -s $file_pred_new_name -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
    fi
  done

else
  echo "Running QC for MONAI predictions ..."
  # Check if file exists (pred file)
  for pred_sub in ${PATH_PRED_SEG}/sub-*; do
    echo $pred_sub
    if [[ ${pred_sub} =~ ${SUBJECT} ]]; then
      echo "Subject found, running QC report $pred_sub"
      # cd ${SUBJECT}
      
      for file_pred in ${PATH_PRED_SEG}/${SUBJECT}/*_pred.nii.gz; do
        file_seg_basename=${file_pred##*/}  # keep only the file name from the whole path
        echo $file_seg_basename
        type=func   # always "func" not "anat" or "dwi"

        # rsync prediction mask
        rsync -avzh $file_pred ${type}/$file_seg_basename
        file_image=${file_seg_basename}
        echo $file_image  # should be same as file_seg_basename
        file_image="${file_image/_pred.nii.gz/}"  # deletes suffix

        pred_seg="${type}/${file_seg_basename}"
        pred_seg_bin="${type}/${file_seg_basename/.nii.gz/_bin.nii.gz}"

        # binarize the prediction with threshold 0.5
        sct_maths -i ${pred_seg} -bin 0.5 -o ${pred_seg_bin}

        # Create QC for pred mask
        sct_qc -i ${type}/${file_image}.nii.gz -s ${pred_seg_bin} -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
      done
    else
      echo "Pred mask not found"
    fi
  done
fi

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
