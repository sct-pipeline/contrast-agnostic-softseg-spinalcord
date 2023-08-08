#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:4          # Request 4 GPU "generic resources”.
#SBATCH --tasks-per-node=8    # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=4 # increase this parameter and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=16G      
#SBATCH --time=0-15:00
#SBATCH --output=%N-%j.out

module load python # Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torchvision pytorch-lightning wandb torch scikit-learn scipy Pillow pandas numpy nibabel monai matplotlib  --no-index

export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

# PyTorch Lightning will query the environment to figure out if it is running inside a SLURM batch job
# If it is, it expects the user to have requested one task per GPU.
# If you do not ask for 1 task per GPU, and you do not run your script with "srun", your job will fail!

srun python main.py -m unet -nspv 4 -ncv 1 -initf 8 -bs 4 -lr 1e-3 -cve 4 -stp -epb -nw 4 -djn "dataset_sg_b_ins_sc_can_fmri_dzu_szu_gms_spar_bav_beijt_ukb_scttest_ws_lumepfl"