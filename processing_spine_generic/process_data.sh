#!/bin/bash
#
# Process data. 
#     For T1w and T2w : From raw images, proceeds to resampling and reorientation to RPI.
#     For T2s : Compute root-mean square across 4th dimension (if it exists)
#     For dwi : Generate mean image after motion correction.
#     
#     Generates soft segmentations.
# Usage:
#   ./process_data.sh <SUBJECT>
#
#
# Authors: Sandrine Bédard

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# Uncomment for full verbose
set -x

# Immediately exit if error
#set -e -o pipefail  comment if qform/sform error

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1
EXCLUDE_LIST=$2

# Save script path
PATH_SCRIPT=$PWD

# get starting time:
start=`date +%s`

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED
# Copy list of participants in processed data folder
if [[ ! -f "participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv .
fi
if [[ ! -f "participants.json" ]]; then
  rsync -avzh $PATH_DATA/participants.json .
fi
if [[ ! -f "dataset_description.json" ]]; then
  rsync -avzh $PATH_DATA/dataset_description.json .
fi
if [[ ! -f "README.md" ]]; then
  rsync -avzh $PATH_DATA/README.md .
fi
# Copy list of participants in results folder
if [[ ! -f $PATH_RESULTS/"participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv $PATH_RESULTS/"participants.tsv"
fi
# Copy source images
rsync -avzh $PATH_DATA/$SUBJECT .


# FUNCTIONS
# ==============================================================================

# If there is an additional b=0 scan, add it to the main DWI data and update the
# bval and bvec files.
concatenate_b0_and_dwi(){
  local file_b0="$1"  # does not have extension
  local file_dwi="$2"  # does not have extension
  if [[ -e ${file_b0}.nii.gz ]]; then
    echo "Found additional b=0 scans: $file_b0.nii.gz They will be concatenated to the DWI scans."
    sct_dmri_concat_b0_and_dwi -i ${file_b0}.nii.gz ${file_dwi}.nii.gz -bval ${file_dwi}.bval -bvec ${file_dwi}.bvec -order b0 dwi -o ${file_dwi}_concat.nii.gz -obval ${file_dwi}_concat.bval -obvec ${file_dwi}_concat.bvec
    # Update global variable
    FILE_DWI="${file_dwi}_concat"
  else
    echo "No additional b=0 scans was found."
    FILE_DWI="${file_dwi}"
  fi
}

# Check if manual label already exists. If it does, copy it locally. If it does
# not, perform labeling.
# NOTE: manual disc labels should go from C1-C2 to C7-T1.
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  local file_space_other="$3"  # image file that has a different resolution than the segmentation file (but still in the same physical space)
  # Update global variable with segmentation file name
  FILELABEL="${file}_label-discs_dlabel" #label-discs_dlabel
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${FILELABEL}.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c t2 -ofolder ./anat/
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation
    sct_label_vertebrae -i ${file_space_other}.nii.gz -s ${file_seg}.nii.gz -c t2 -ofolder ./anat/
  fi
}


find_manual_seg(){
  local file="$1"
  local contrast="$2"
  local contrast_for_seg="$3"
  # Find contrast
  if [[ $contrast == "./dwi/" ]]; then
    folder_contrast="dwi"
  else
    folder_contrast="anat"
  fi

  # Update global variable with segmentation file name
  FILESEG="${file}_label-SC_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${folder_contrast}/${FILESEG}.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL "${folder_contrast}/${FILESEG}.nii.gz"
    sct_qc -i ${folder_contrast}/${file}.nii.gz -s ${folder_contrast}/${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Manual segmentation not found."
    # Segment spinal cord
    sct_deepseg_sc -i ${folder_contrast}/${file}.nii.gz -c $contrast_for_seg -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${folder_contrast}/${FILESEG}.nii.gz
    
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

yaml() {
    python3 -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
}

# Go to anat folder where all structural data are located
cd ${SUBJECT}/anat/

# T1w
# ------------------------------------------------------------------------------

file_t1="${SUBJECT}_T1w"
# Check if T1w image exists
if [[ -f ${file_t1}.nii.gz ]];then

  # Rename the raw image
  mv ${file_t1}.nii.gz ${file_t1}_raw.nii.gz
  file_t1="${file_t1}_raw"

  # Reorient to RPI and resample to 1 mm iso (supposed to be the effective resolution)
  sct_image -i ${file_t1}.nii.gz -setorient RPI -o ${file_t1}_RPI.nii.gz
  sct_resample -i ${file_t1}_RPI.nii.gz -mm 1x1x1 -o ${file_t1}_RPI_r.nii.gz
  file_t1="${file_t1}_RPI_r"

  # Rename _RPI_r file
  mv ${file_t1}.nii.gz ${SUBJECT}_space-other_T1w.nii.gz
  file_t1="${SUBJECT}_space-other_T1w"
  # Delete raw and reoriented to RPI images
  mv ${SUBJECT}_T1w_raw.nii.gz ${SUBJECT}_T1w.nii.gz
  rm -f ${SUBJECT}_T1w_raw_RPI.nii.gz
fi

# T2
# ------------------------------------------------------------------------------
file_t2="${SUBJECT}_T2w"
# Check if T2w image exists
if [[ -f ${file_t2}.nii.gz ]];then
  # Rename raw file
  mv ${file_t2}.nii.gz ${file_t2}_raw.nii.gz
  file_t2="${file_t2}_raw"

  # Reorient to RPI and resample to 0.8mm iso (supposed to be the effective resolution)
  sct_image -i ${file_t2}.nii.gz -setorient RPI -o ${file_t2}_RPI.nii.gz
  sct_resample -i ${file_t2}_RPI.nii.gz -mm 0.8x0.8x0.8 -o ${file_t2}_RPI_r.nii.gz
  file_t2="${file_t2}_RPI_r"

  # Rename _RPI_r file
  mv ${file_t2}.nii.gz "${SUBJECT}_space-other_T2w.nii.gz"
  file_t2="${SUBJECT}_space-other_T2w"
  # Delete raw, reoriented to RPI images
  rm -f ${SUBJECT}_T2w_raw_RPI.nii.gz
  mv ${SUBJECT}_T2w_raw.nii.gz ${SUBJECT}_T2w.nii.gz
fi

# T2s
# ------------------------------------------------------------------------------
file_t2s="${SUBJECT}_T2star"
# Check if T2star image exists
if [[ -f ${file_t2s}.nii.gz ]];then
  # Rename raw file
  mv ${file_t2s}.nii.gz ${file_t2s}_raw.nii.gz
  file_t2s="${file_t2s}_raw"

  # Compute root-mean square across 4th dimension (if it exists), corresponding to all echoes in Philips scans.
  sct_maths -i ${file_t2s}.nii.gz -rms t -o ${file_t2s}_rms.nii.gz
  file_t2s="${file_t2s}_rms"

  # Rename _rms file
  mv ${file_t2s}.nii.gz "${SUBJECT}_space-other_T2star.nii.gz"
  file_t2s="${SUBJECT}_space-other_T2star"
  # Delete raw images
  rm -f ${SUBJECT}_T2star_raw.nii.gz
fi

# MTS
# ------------------------------------------------------------------------------
file_t1w="${SUBJECT}_flip-2_mt-off_MTS"
if [[ -f ${file_t1w}.nii.gz ]];then
  # Rename raw file
  mv ${file_t1w}.nii.gz ${file_t1w}_raw.nii.gz
  file_t1w="${file_t1w}_raw"

  # Reorient to RPI
  sct_image -i ${file_t1w}.nii.gz -setorient RPI -o ${file_t1w}_RPI.nii.gz
  file_t1w="${file_t1w}_RPI"

  # Rename _RPI file
  mv ${file_t1w}.nii.gz "${SUBJECT}_flip-2_mt-off_space-other_MTS.nii.gz"
  file_t1w="${SUBJECT}_flip-2_mt-off_space-other_MTS"
  # Delete raw
  rm -f ${SUBJECT}_flip-2_mt-off_MTS_raw.nii.gz
fi


file_mton="${SUBJECT}_flip-1_mt-on_MTS"
if [[ -f ${file_mton}.nii.gz ]];then
  # Rename raw file
  mv ${file_mton}.nii.gz ${file_mton}_raw.nii.gz
  file_mton="${file_mton}_raw"

  # Reorient to RPI
  sct_image -i ${file_mton}.nii.gz -setorient RPI -o ${file_mton}_RPI.nii.gz
  file_mton="${file_mton}_RPI"

  # Rename _RPI file
  mv ${file_mton}.nii.gz "${SUBJECT}_flip-1_mt-on_space-other_MTS.nii.gz"
  file_mton="${SUBJECT}_flip-1_mt-on_space-other_MTS"
  # Delete raw
  rm -f ${SUBJECT}_flip-1_mt-on_MTS_raw.nii.gz
fi


cd ..
# DWI
# ------------------------------------------------------------------------------
file_dwi="${SUBJECT}_dwi"
# Check if dwi images exists
if [[ -f ./dwi/${file_dwi}.nii.gz ]];then
  cd ./dwi
  # If there is an additional b=0 scan, add it to the main DWI data
  concatenate_b0_and_dwi "${SUBJECT}_acq-b0_dwi" $file_dwi
  file_dwi=$FILE_DWI
  file_bval=${file_dwi}.bval
  file_bvec=${file_dwi}.bvec
  # Separate b=0 and DW images
  sct_dmri_separate_b0_and_dwi -i ${file_dwi}.nii.gz -bvec ${file_bvec}
  # Get centerline
  sct_get_centerline -i ${file_dwi}_dwi_mean.nii.gz -c dwi -qc ${PATH_QC} -qc-subject ${SUBJECT}
  # Create mask to help motion correction and for faster processing
  sct_create_mask -i ${file_dwi}_dwi_mean.nii.gz -p centerline,${file_dwi}_dwi_mean_centerline.nii.gz -size 30mm
  # Motion correction
  sct_dmri_moco -i ${file_dwi}.nii.gz -bvec ${file_dwi}.bvec -m mask_${file_dwi}_dwi_mean.nii.gz -x spline

  # Rename _moco_dwi_mean file
  mv ${FILE_DWI}_moco_dwi_mean.nii.gz ${SUBJECT}_rec-average_dwi.nii.gz
  file_dwi_mean="${SUBJECT}_rec-average_dwi"
  # Remove intermediate files
  if [[ -e ${SUBJECT}_acq-b0_dwi.nii.gz ]]; then
    rm -f mask_${FILE_DWI}_dwi_mean.nii.gz moco_params.tsv moco_params_x.nii.gz moco_params_y.nii.gz ${FILE_DWI}.bval ${FILE_DWI}.bvec ${FILE_DWI}.nii.gz ${FILE_DWI}_b0.nii.gz ${FILE_DWI}_b0_mean.nii.gz ${FILE_DWI}_dwi.nii.gz ${FILE_DWI}_dwi_mean.nii.gz ${FILE_DWI}_dwi_mean_centerline.nii.gz ${FILE_DWI}_moco.nii.gz ${FILE_DWI}_moco_b0_mean.nii.gz ${FILE_DWI}_dwi_mean_centerline.csv
  else
    rm -f mask_${FILE_DWI}_dwi_mean.nii.gz moco_params.tsv moco_params_x.nii.gz moco_params_y.nii.gz ${FILE_DWI}_b0.nii.gz ${FILE_DWI}_b0_mean.nii.gz ${FILE_DWI}_dwi.nii.gz ${FILE_DWI}_dwi_mean.nii.gz ${FILE_DWI}_dwi_mean_centerline.nii.gz ${FILE_DWI}_moco.nii.gz ${FILE_DWI}_moco_b0_mean.nii.gz ${FILE_DWI}_dwi_mean_centerline.csv
  fi
  cd ..
fi


# Initialize filenames
contrasts=($file_t1 $file_t2s $file_t1w $file_mton $file_dwi_mean)
inc_contrasts=()

# Check if a list of images to exclude was passed.
if [ -z "$EXCLUDE_LIST" ]; then
  EXCLUDE=""
else
  EXCLUDE=$(yaml $PATH_SCRIPT/${EXCLUDE_LIST} "['FILES_REG']")
fi

for contrast in "${contrasts[@]}"; do
  type=$(find_contrast $contrast)
  if echo "$EXCLUDE" | grep -q "$contrast"; then
    echo "$contrast found in exclude list.";
  else
    if [[ -f "${type}${contrast}.nii.gz" ]]; then
      inc_contrasts+=(${type}${contrast})
    else
      echo "$contrast not found, excluding it."
    fi
  fi

done
echo "Contrasts are" ${inc_contrasts[@]}

# Generate softsegs
# Create mask for regsitration
find_manual_seg ${file_t2} 'anat' 't2'
file_t2_seg="${file_t2}_seg"
file_t2_mask="${file_t2_seg}_mask"
sct_create_mask -i ./anat/${file_t2}.nii.gz -p centerline,./anat/${file_t2_seg}.nii.gz -size 55mm -o ./anat/${file_t2_mask}.nii.gz 

# Label intervertebral discs of T2w
label_if_does_not_exist ./anat/${SUBJECT}_T2w ./anat/${file_t2_seg} ./anat/${file_t2}
file_t2_seg_labeled="${file_t2_seg}_labeled"
file_t2_discs="${file_t2_seg}_labeled_discs"

# Generate QC report to assess vertebral labeling
sct_qc -i ./anat/${file_t2}.nii.gz -s ./anat/${file_t2_seg_labeled}.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Loop through available contrasts
for file_path in "${inc_contrasts[@]}";do
  # Find contrast to do segmentation
  if [[ $file_path == *"T1w"* ]];then
      contrast_seg="t1"
      contrast="t1w"
  elif [[ $file_path == *"T2star"* ]];then
      contrast_seg="t2s"
      contrast=contrast_seg
  elif [[ $file_path == *"flip-2_mt-off_space-other_MTS"* ]];then
      contrast_seg="t1"
      contrast="flip-2_mt-off_space-other__MTS"
  elif [[ $file_path == *"flip-1_mt-on_space-other_MTS"* ]];then
      contrast_seg="t2s"
      contrast="flip-1_mt-on_space-other_MTS"
  elif [[ $file_path == *"dwi"* ]];then
      contrast_seg="dwi"
      contrast=contrast_seg
  fi

  type=$(find_contrast $file_path)
  file=${file_path/#"$type"}
  fileseg=${file_path}_seg
  find_manual_seg $file $type $contrast_seg

  # Add padding to seg to overcome edge effect
  python ${PATH_SCRIPT}/pad_seg.py -i ${fileseg}.nii.gz -o ${fileseg}_pad.nii.gz
  # Registration
  # ------------------------------------------------------------------------------
  sct_register_multimodal -i ${file_path}.nii.gz -d ./anat/${file_t2}.nii.gz -iseg ${fileseg}_pad.nii.gz -dseg ./anat/${file_t2_seg}.nii.gz -param step=1,type=seg,algo=centermass -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${file_path}_reg.nii.gz
  warping_field=${type}warp_${file}2${file_t2}

  # Generate SC segmentation coverage and register to T2w
  # ------------------------------------------------------------------------------
  sct_create_mask -i ${file_path}.nii.gz -o ${file_path}_ones.nii.gz -size 500 -p centerline,${fileseg}.nii.gz
  # Bring coverage mask to T2w space
  sct_apply_transfo -i ${file_path}_ones.nii.gz -d ./anat/${file_t2}.nii.gz -w ${warping_field}.nii.gz -x linear -o ${file_path}_ones_reg.nii.gz
  # Bring SC segmentation to T2w space
  sct_apply_transfo -i ${fileseg}.nii.gz -d ./anat/${file_t2_seg}.nii.gz -w ${warping_field}.nii.gz -x linear -o ${fileseg}_reg.nii.gz
  # Remove 8 to 10 slices before adding all segmentation because of partial slices (except for T1w)
  if [[ $contrast != "t1w" ]];then
        mv ${fileseg}_reg.nii.gz ${fileseg}_reg_no_crop.nii.gz
        mv ${file_path}_ones_reg.nii.gz ${file_path}_ones_reg_no_crop.nii.gz
        python ${PATH_SCRIPT}/remove_slices_seg.py -i ${fileseg}_reg_no_crop.nii.gz -c contrast -coverage-map ${file_path}_ones_reg_no_crop.nii.gz -o ${fileseg}_reg.nii.gz -o-coverage-map ${file_path}_ones_reg.nii.gz
  fi

done
# Create coverage mask for T2w
sct_create_mask -i ./anat/${file_t2}.nii.gz -o ./anat/${file_t2}_ones.nii.gz -size 500 -p centerline,./anat/${file_t2_seg}.nii.gz
# Create soft SC segmentation
# ------------------------------------------------------------------------------
# Sum all coverage images
contrasts_coverage=${inc_contrasts[@]/%/"_ones_reg.nii.gz"}
sct_maths -i ./anat/${file_t2}_ones.nii.gz -add $(eval echo ${contrasts_coverage[@]}) -o sum_coverage.nii.gz
# Sum all segmentations
contrasts_seg=${inc_contrasts[@]/%/"_seg_reg.nii.gz"}
sct_maths -i ./anat/${file_t2_seg}.nii.gz -add $(eval echo ${contrasts_seg[@]}) -o sum_sc_seg.nii.gz
# Divide sum_sc_seg by sum_coverage
sct_maths -i sum_sc_seg.nii.gz -div sum_coverage.nii.gz -o ./anat/${file_t2}_seg_soft.nii.gz

file_softseg=./anat/"${file_t2}_seg_soft"
# Check if softseg has NaN values, if so, change to 0
python ${PATH_SCRIPT}/check_if_nan.py -i ${file_softseg}.nii.gz -o ./anat/${file_t2}_softseg.nii.gz

# Create QC report of softseg on T2w
sct_qc -i ./anat/${file_t2}.nii.gz -s ${file_softseg}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Bring back softseg to native space and generate QC report 
# ------------------------------------------------------------------------------
for file_path in "${inc_contrasts[@]}";do
  type=$(find_contrast $file_path)
  file=${file_path/#"$type"}
  fileseg=${file_path}_seg
  warping_field_inv=${type}warp_${file_t2}2${file}

  # Bring softseg to native space
  sct_apply_transfo -i ${file_softseg}.nii.gz -d ${file_path}.nii.gz -w ${warping_field_inv}.nii.gz -x linear -o ${file_path}_softseg.nii.gz
  # Apply coverage mask to softseg
  sct_maths -i ${file_path}_softseg.nii.gz -o ${file_path}_softseg.nii.gz -mul ${file_path}_ones.nii.gz

  # Generate QC report
  sct_qc -i ${file_path}.nii.gz -s ${file_path}_softseg.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  
  # For T1w contrast, use existing disc file labels
  if [[ $file_path == *"T1w"* ]];then
    label_if_does_not_exist ./anat/${SUBJECT_T1w} ${fileseg} ${file_path} 
  else
    # Bring T2w disc labels to native space
    sct_apply_transfo -i ./anat/${file_t2_discs}.nii.gz -d ${file_path}.nii.gz -w ${warping_field_inv}.nii.gz -x label -o ${file_path}_seg_labeled_discs.nii.gz
    # Set sform to qform (there are disparencies)
    sct_image -i ${file_path}_seg_labeled_discs.nii.gz -set-sform-to-qform
    sct_image -i ${fileseg}.nii.gz -set-sform-to-qform
    sct_image -i ${file_path}.nii.gz -set-sform-to-qform
    # Generate labeled segmentation from warp disc labels
    sct_label_vertebrae -i ${file_path}.nii.gz -s ${fileseg}.nii.gz -discfile ${file_path}_seg_labeled_discs.nii.gz -c t2 -ofolder $type
  fi
  # Generate QC report to assess vertebral labeling
  sct_qc -i ${file_path}.nii.gz -s ${fileseg}_labeled.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}

done


# Go back to root output folder
cd $PATH_OUTPUT
# Create and populate clean data processed folder for training
PATH_DATA_PROCESSED_CLEAN="${PATH_DATA_PROCESSED}_clean"
# Copy over required BIDs files
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/participants.* $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/README.md $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/

inc_contrasts+=("./anat/${file_t2}")
for file_path in "${inc_contrasts[@]}";do
  type=$(find_contrast $file_path)
  file=${file_path/#"$type"}
  fileseg=${file_path}_label-SC_seg
  filesoftseg=${file_path}_label-SC_softseg
  fileseglabel=${file_path}_label-SC_seg_labeled
  fileseglabel_bin=${file_path}_desc-softseg_label-SC_seg
  cd $PATH_DATA_PROCESSED/$SUBJECT 

  # Find contrast to name csa files
  if [[ $file_path == *"flip-2_mt-off_space-other_MTS"* ]];then
      contrast_seg="flip-2_mt-off_space-other_MTS"
  elif [[ $file_path == *"T1w"* ]];then
      contrast_seg="T1w"
  elif [[ $file_path == *"T2w"* ]];then
      contrast_seg="T2w"
  elif [[ $file_path == *"T2star"* ]];then
      contrast_seg="T2star"
  elif [[ $file_path == *"flip-1_mt-on_space-other_MTS"* ]];then
      contrast_seg="flip-1_mt-on_space-other_MTS"
  elif [[ $file_path == *"dwi"* ]];then
      contrast_seg="dwi"
  fi
  # Clip softsegs
  python ${PATH_SCRIPT}/clip_softseg.py -i ${filesoftseg}.nii.gz -o ${filesoftseg}.nii.gz

  # Compute CSA on soft GT and hard GT (only from the derivaives)
  # Soft segmentation
  sct_process_segmentation -i ${PATH_DATA}/derivatives/labels_softseg/${SUBJECT}/${filesoftseg}.nii.gz -vert 2,3 -vertfile ${fileseglabel}.nii.gz -o ${PATH_RESULTS}/csa_soft_GT_${contrast_seg}.csv -append 1
  # Soft segmentation binarized
  # Binarize output softseg for CSA computation
  sct_maths -i ${PATH_DATA}/derivatives/labels_softseg/${SUBJECT}/${filesoftseg}.nii.gz -bin 0.5 -o ${filesoftseg}_bin.nii.gz
  # Create QC report
  sct_qc -i $PATH_DATA_PROCESSED/${SUBJECT}/${file_path}.nii.gz -s ${filesoftseg}_bin.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  sct_process_segmentation -i ${filesoftseg}_bin.nii.gz -vert 2,3 -vertfile ${fileseglabel}.nii.gz -o ${PATH_RESULTS}/csa_soft_GT_bin_${contrast_seg}.csv -append 1

  # Hard segmentation
  sct_process_segmentation -i ${PATH_DATA}/derivatives/labels/${SUBJECT}/${fileseg}.nii.gz -vert 2,3 -vertfile ${fileseglabel}.nii.gz -o ${PATH_RESULTS}/csa_hard_GT_${contrast_seg}.csv -append 1
  
  mkdir -p $PATH_DATA_PROCESSED_CLEAN $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/$type $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/$type
  mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg/${SUBJECT}/$type
  mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg_bin/${SUBJECT}/$type  # Create labels_softseg_bin folder

  # Put image in cleaned dataset
  rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/${file_path}.nii.gz $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/${file_path}.nii.gz

  # Don't copy .json file 
  if echo "$file_path" | grep -q "./dwi"; then
    echo "No .json file for dwi."
  else  
    rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/${file_path}.json $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/${file_path}.json
  fi

  # Move segmentation and soft segmentation to the cleanded derivatives
  rsync -avzh ${PATH_DATA}/derivatives/labels/${SUBJECT}/${fileseg}.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/${fileseg}.nii.gz
  rsync -avzh ${PATH_DATA}/derivatives/labels_softseg/${SUBJECT}/${filesoftseg}.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg/${SUBJECT}/${filesoftseg}.nii.gz
  rsync -avzh ${filesoftseg}_bin.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg_bin/${SUBJECT}/${fileseglabel_bin}.nii.gz
  # Move json files of derivatives
  rsync -avzh "${PATH_DATA}/derivatives/labels/${SUBJECT}/${fileseg}.json" $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/${fileseg}.json
  rsync -avzh "${PATH_DATA}/derivatives/labels_softseg/${SUBJECT}/${filesoftseg}.json" $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg/${SUBJECT}/${filesoftseg}.json
  rsync -avzh "${PATH_DATA}/derivatives/labels_softseg/${SUBJECT}/${filesoftseg}.json" $PATH_DATA_PROCESSED_CLEAN/derivatives/labels_softseg_bin/${SUBJECT}/${fileseglabel_bin}.json
 
  # Move disc labels into cleaned derivatives
  # TODO: ADD IF STATEMENT FOR T1 and T2w label-discs_dlabel instead
  if [[ $contrast_seg == *"T2w"* || $contrast_seg == *"T1w"* ]];then
    rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/${file_path}_seg_labeled_discs.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/${file_path}_label-discs_dlabel.nii.gz
  else
    rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/${file_path}_seg_labeled_discs.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/${file_path}_label-discs_desc-warp_dlabel.nii.gz
  fi
done



# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
FILES_TO_CHECK=(
  "anat/${SUBJECT}_space-other_T1w.nii.gz"
  "anat/${SUBJECT}_space-other_T2w.nii.gz"
  "anat/${SUBJECT}_space-other_T2star.nii.gz"
  "dwi/${SUBJECT}_rec-average_dwi.nii.gz"
)
pwd
for file in ${FILES_TO_CHECK[@]}; do
  if [[ ! -e $file ]]; then
    echo "${SUBJECT}/anat/${file} does not exist" >> $PATH_LOG/_error_check_output_files.log
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
