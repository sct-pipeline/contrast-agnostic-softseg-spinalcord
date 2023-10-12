#!/bin/bash
#
# Compute CSA on output segmentations. Input is data_processed_clean
# Adapted from compute_csa_nnunet.sh
#
# Usage:
#   ./compute_csa.sh <SUBJECT>
#
# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/anat/
#
# Authors: Naga Karthik 

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
rsync -avzh $PATH_DATA/$SUBJECT .

# FUNCTIONS
# ==============================================================================

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



# SCRIPT STARTS HERE
# ==============================================================================
# Go to anat folder where all structural data are located
cd ${SUBJECT}
file_sub="${SUBJECT//[\/]/_}"

# Check if file exists (pred file)
for pred_sub in ${PATH_PRED_SEG}/sub-*; do
  if [[ ${pred_sub} =~ ${SUBJECT} ]]; then
    echo "Subject found, running QC report $pred_sub"
    # cd ${SUBJECT}
    
    for file_pred in ${PATH_PRED_SEG}/${SUBJECT}/*_pred.nii.gz; do
      file_seg_basename=${file_pred##*/}  # keep only the file name from the whole path
      echo $file_seg_basename
      if [[ ${file_seg_basename} =~ "dwi" ]]; then
        type=dwi
      else
        type=anat
      fi

      # rsync prediction mask
      rsync -avzh $file_pred ${type}/$file_seg_basename
      file_image=${file_seg_basename}
      echo $file_image  # should be same as file_seg_basename
      file_image="${file_image/_pred.nii.gz/}"  # deletes suffix
      # Get contrast for CSV file (e.g., T1w, flip-1_mt-on_MTS, dwi, ...)
      contrast_csv=${file_image#*_}
      contrast_csv=${contrast_csv/rec-average_/}  # remove rec-average from dwi 

      pred_seg="${type}/${file_seg_basename}"
      pred_seg_bin="${type}/${file_seg_basename/.nii.gz/_bin.nii.gz}"

      # NOTE: soft QC is still buggy so binarize the prediction with threshold 0.5
      sct_maths -i ${pred_seg} -bin 0.5 -o ${pred_seg_bin}

      # Create QC for pred mask
      # NOTE: this is raising errors for subjects - sub-stanford06 and sub-cmrrb01
      sct_qc -i ${type}/${file_image}.nii.gz -s ${pred_seg_bin} -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
           

      # Get manual hard GT to get labeled segmentation
      FILESEG="${type}/${file_image}_seg" 
      FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${FILESEG}-manual.nii.gz"
      echo
      echo "Looking for manual segmentation: $FILESEGMANUAL"
      if [[ -e $FILESEGMANUAL ]]; then
        echo "Found! Using manual segmentation."
        rsync -avzh $FILESEGMANUAL "${FILESEG}.nii.gz"
      fi
      # Create labeled segmentation of vertebral levels (only if it does not exist) 
      label_if_does_not_exist ${type}/${file_image} $FILESEG $type

      file_seg_labeled="${FILESEG}_labeled"
      # Generate QC report to assess vertebral labeling
      sct_qc -i ${type}/${file_image}.nii.gz -s ${file_seg_labeled}.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}

      # Compute average cord CSA between C2 and C3 (on soft predictions)
      sct_process_segmentation -i ${pred_seg} -vert 2:3 -vertfile ${file_seg_labeled}.nii.gz -o ${PATH_RESULTS}/csa_pred_${contrast_csv}_soft.csv -append 1

      # Compute average cord CSA between C2 and C3 (on binarized predictions)
      sct_process_segmentation -i ${pred_seg_bin} -vert 2:3 -vertfile ${file_seg_labeled}.nii.gz -o ${PATH_RESULTS}/csa_pred_${contrast_csv}_soft_bin.csv -append 1

    done
  
  else
    echo "Pred mask not found"
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
