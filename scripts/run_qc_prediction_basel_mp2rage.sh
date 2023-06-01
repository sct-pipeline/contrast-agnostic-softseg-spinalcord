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
        #rsync -avzh $file_pred ${type}/$file_seg_basename
        prefix="baselMP2RageRPI_"  #TODO  
        file_image=${file_seg_basename#"$prefix"}
        echo $file_image
        file_image="${file_image::-11}"  # Remove X.nii.gz since the number X varies
        # split with "-"
        arrIN=(${file_image//-/ })
        if [[ $type == *"dwi"* ]];then
          contrast="rec-average_dwi"  # original image name
        else
          contrast=${file_image#${arrIN[0]}"-"}  # remove sub-
          contrast=${contrast#${arrIN[1]}"-"}  # remove sub-id
	  contrast="_UNIT1"
        fi
        file_image=${arrIN[0]}"-"${arrIN[1]}"_"${contrast}".nii.gz"
	file_image=${arrIN[0]}"-"${arrIN[1]}"${contrast}.nii.gz"
        echo $file_image
	# rsync prediction mask
	file_pred_new_name=${type}/${arrIN[0]}"-"${arrIN[1]}"${contrast}_pred.nii.gz"
	rsync -avzh $file_pred $file_pred_new_name
	# Create QC for pred mask
	sct_qc -i ${type}/${file_image} -s $file_pred_new_name -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}


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
