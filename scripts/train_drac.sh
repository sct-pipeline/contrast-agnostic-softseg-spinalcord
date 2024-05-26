#!/bin/bash
# 
# This script is used for training the contrast-agnostic model on DRAC/Compute Canada:
# 1. Creates a virtual environment `venv_monai` and installs the required dependencies
# 2. Generates a MSD-style datalist containing image/label pairs for training
# 3. Trains the contrast-agnostic soft segmentation model
# 4. Evaluates the model on the test set
# 
# Usage:
# bash train.sh <path_to_preprocessed_data> <contrast> <label_type> <path_to_train_yaml>
# 
# Examples:
# 1. Train a model on 'T1w' contrast with 'hard' labels
#       bash train.sh /path/to/spine-generic-processed/ t1w hard train.yaml
# 2. Train a model on 'all' contrasts with 'soft' labels
#       bash train.sh /path/to/spine-generic-processed/ all soft train.yaml
#
#


# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# load the required modules
echo "Loading modules ..."
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2 
# trying to see if cupy is able to fetch libnvrtc.so.12
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$EBROOTCUDA/lib64/"

# activate environment
echo "Activating environment ..."
source /home/$(whoami)/envs/venv_monai/bin/activate

# Set the following variables to the desired values
PATH_REPO="/home/$(whoami)/code/contrast-agnostic/contrast-agnostic-softseg-spinalcord"

datalist_out_dir="model_v25_11datasets"
DATASETS_ROOT="/home/$(whoami)/projects/rrg-bengioy-ad/$(whoami)/datasets"
PATH_DATASPLITS="/home/$(whoami)/code/contrast-agnostic/datasplits"
PATH_DATALIST_ROOT="/home/$(whoami)/code/contrast-agnostic/datalists/${datalist_out_dir}"
if [[ ! -d $PATH_DATALIST_ROOT ]]; then
    mkdir $PATH_DATALIST_ROOT
    mkdir $PATH_DATALIST_ROOT/logs
fi

# List of datasets to train on
DATASETS=("basel-mp2rage" "canproco" "data-multi-subject" "dcm-zurich" "lumbar-epfl" "lumbar-vanderbilt" "nih-ms-mp2rage" "sci-colorado" "sci-paris" "sci-zurich" "sct-testing-large")

PATH_RESULTS="/home/$(whoami)/code/contrast-agnostic/results"
PATH_CONFIG="${PATH_REPO}/configs/train_all.yaml"

echo "-------------------------------------------"
echo "Moving the datasets to SLURM_TMPDIR: ${SLURM_TMPDIR}"
echo "-------------------------------------------"

# create folders in SLURM_TMPDIR
if [[ ! -d $SLURM_TMPDIR/datasets ]]; then
    mkdir $SLURM_TMPDIR/datasets
fi

if [[ ! -d $SLURM_TMPDIR/datalists ]]; then
    mkdir $SLURM_TMPDIR/datalists
fi

if [[ ! -d $SLURM_TMPDIR/results ]]; then
    mkdir $SLURM_TMPDIR/results
fi

# get starting time:
data_start=`date +%s`

for dataset in ${DATASETS[@]}; do
    if [[ ! -d $SLURM_TMPDIR/datasets/$dataset ]]; then
        # echo 
        echo "Copying ${dataset} to ${SLURM_TMPDIR}/datasets ..."
        
        # copy the dataset to SLURM_TMPDIR
        # NOTE cp is copying also the .git files and git-annex SHA files which are not needed
        # cp -r ${DATASETS_ROOT}/$dataset ${SLURM_TMPDIR}/datasets
        rsync -azh --exclude=".*" ${DATASETS_ROOT}/$dataset ${SLURM_TMPDIR}/datasets
    fi

    echo "-------------------------------------------------------"
    echo "Creating datalist for ${dataset} ..."
    echo "-------------------------------------------------------"

    # create the datalist json file
    python ${PATH_REPO}/monai/create_msd_data_cc.py \
        --path-data $SLURM_TMPDIR/datasets/$dataset \
        --path-datasplit ${PATH_DATASPLITS}/datasplit_${dataset}_seed50.yaml \
        --path-out ${SLURM_TMPDIR}/datalists \
        --seed 50
done

echo "-------------------------------------------------------"
echo "Copying the datalists back to: ${PATH_DATALIST_ROOT} ..."
echo "-------------------------------------------------------"
# copy the datalist to the datalist root
cp ${SLURM_TMPDIR}/datalists/*.json ${PATH_DATALIST_ROOT}
cp ${SLURM_TMPDIR}/datalists/*.csv ${PATH_DATALIST_ROOT}
cp ${SLURM_TMPDIR}/datalists/logs/*.txt ${PATH_DATALIST_ROOT}/logs

echo "-------------------------------------------------------"
echo "Datalist creation done! Starting model training ..."
echo "-------------------------------------------------------"

data_end=`date +%s`
runtime=$((data_end-data_start))
echo "-------------------------------------------------------"
echo "Moving datasets to SLURM_TMPDIR took: $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "-------------------------------------------------------"

# enable wandb offline mode
wandb offline

train_start=`date +%s`

# Train the model
# srun python ${PATH_REPO}/monai/main.py \
srun torchrun --standalone --nnodes=$SLURM_JOB_NUM_NODES --nproc-per-node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    ${PATH_REPO}/monai/main.py \
    --path-datalists ${SLURM_TMPDIR}/datalists \
    --path-results ${SLURM_TMPDIR}/results \
    --model 'nnunet' \
    --config $PATH_CONFIG \
    --pad-mode zero \
    --input-label 'soft' \
    --enable-pbar \

echo "-------------------------------------------------------"
echo "Model training done! Copying the results back to home ..."
echo "-------------------------------------------------------"

train_end=`date +%s`
train_runtime=$((train_end-train_start))
echo "-------------------------------------------------------"
echo "Total Training Time: $(($train_runtime / 3600))hrs $((($train_runtime / 60) % 60))min $(($train_runtime % 60))sec"
echo "-------------------------------------------------------"

# copy the results back to the respective directories on home
cp -r $SLURM_TMPDIR/results/* $PATH_RESULTS
