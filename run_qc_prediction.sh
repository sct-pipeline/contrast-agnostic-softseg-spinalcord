#!/bin/bash
#
# Run QC report on prediction masks
# Usage:
#   ./run_qc_prediction.sh <SUBJECT>
#
#
# Authors: Sandrine Bédard
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# EXAMPLE of command:
# sct_run_batch -jobs 20 -path-data ~/data_nvme_sebeda/datasets/dcm-zurich/ 
#  -path-output ~/data_nvme_sebeda/qc_dcm_zurich -script run_qc_prediction.sh 
# -script-args ~/duke/temp/muena/contrast-agnostic/pure-inference/Dataset721_dcmZurichAxial_test/

set -x
# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1
PATH_PRED_SEG=$2


# get starting time:
start=`date +%s`

# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd ${PATH_DATA_PROCESSED}
# Copy list of participants in processed data folder
if [[ ! -f "participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv .
fi
# Copy list of participants in resutls folder
if [[ ! -f $PATH_RESULTS/"participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv $PATH_RESULTS/"participants.tsv"
fi

# Copy source images
# Copy session if specified
if [[ $SUBJECT == *"ses"* ]];then
    rsync -Ravzh $PATH_DATA/./$SUBJECT .
else
    rsync -avzh $PATH_DATA/$SUBJECT .
fi


# FUNCTIONS
# ==============================================================================

find_contrast(){
  local file="$1"
  local dwi="dwi"
  if echo "$file" | grep -q "$dwi"; then
    echo  "./${dwi}/"
  else
    echo "./anat/"
  fi
}


# SCRIPT STARTS HERE
# ==============================================================================
# Go to anat folder where all structural data are located
cd ${SUBJECT}
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_sub="${SUBJECT//[\/]/_}"

for file_pred in ${PATH_PRED_SEG}/*; do
    if [[ $file_pred == *$file_sub* ]];then
        echo " File found, running QC report $file_pred"
        # Find if anat or dwi
        file_seg_basename=${file_pred##*/}
        echo $file_seg_basename
        type=$(find_contrast $file_pred)

        # rsync prediction mask
        rsync -avzh $file_pred ${type}/$file_seg_basename
        prefix="dcmZurichAxial"  #TODO  
        file_image=${file_seg_basename#"$prefix"}
        echo $file_image
        file_image="${file_image::-11}"  # Remove X.nii.gz since teh number X varies
        # split with "-"
        arrIN=(${file_image//-/ })
        if [[ $type == *"dwi"* ]];then
          contrast="rec-average_dwi"  # original image name
        else
          contrast=${file_image#${arrIN[0]}"-"}  # remove sub-
          contrast=${contrast#${arrIN[1]}"-"}  # remove sub-id
        fi
        file_image=${arrIN[0]}"-"${arrIN[1]}"_"${contrast}".nii.gz"
        echo $file_image
        # Create QC for pred mask
        sct_qc -i ${type}/${file_image} -s ${type}/${file_seg_basename} -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}


    fi
done

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
