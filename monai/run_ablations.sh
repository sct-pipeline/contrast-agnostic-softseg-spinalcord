#!/bin/bash

# This script runs the ablation study on the different contrasts
# for the contrast-agnostic model with soft segmentation labels.


# ablations=("t2w_t1w_dwi_mtoff_t2star" "t2w_t1w_dwi_mtoff" "t2w_t1w_dwi" "t2w_t1w" "t2w")
ablations=("t2w_t1w_dwi_t2star")
cuda_device=3

for ablation in "${ablations[@]}"; do

    echo ""
    echo "-------------------------------------------------"
    echo "Running ablation on contrast(s): ${ablation}"
    echo "-------------------------------------------------"

    # run the model
    CUDA_VISIBLE_DEVICES=${cuda_device} python monai/main.py \
        -r /home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/datalists/spine-generic-contrast-ablations/seed15 \
        -m nnunet -crop 64x192x320 \
        --contrast ${ablation} \
        --label-type soft \
        -initf 32 -me 200 -bs 2 -opt adam -lr 1e-3 -cve 5 -pat 20 \
        -epb --enable_DS \
        --loss AdapW
    
    echo "-------------------------------------------------"
    echo "Finished training!"
    echo "-------------------------------------------------"
done

# Run inference on the remaining contrasts using hard segmentation labels
# unseen_contrasts=("dwi" "mtoff" "mton" "t2star")
unseen_contrasts=("mtoff" "mton")

for contrast in "${unseen_contrasts[@]}"; do

    echo ""
    echo "-------------------------------------------------"
    echo "Running inference on contrast: ${contrast}" using hard seg masks
    echo "-------------------------------------------------"

    # run the model
    CUDA_VISIBLE_DEVICES=${cuda_device} python monai/run_inference.py \
        --path-json ~/contrast-agnostic/datalists/spine-generic/seed15/dataset_${contrast}_hard_seed15.json \
        --chkp-path ~/contrast-agnostic/r1_revision_exps/saved_models/nnunet_t2w_t1w_dwi_t2star_soft_nf=32_opt=adam_lr=0.001_AdapW_bs=2_64x192x320_20240710-2256 \
        --path-out ~/contrast-agnostic/r1_revision_exps/results/nnunet_t2w_t1w_dwi_t2star_soft_nf=32_opt=adam_lr=0.001_AdapW_bs=2_64x192x320_20240710-2256/preds_${contrast} \
        --model nnunet \
        -crop 64x192x-1     
    
    echo "-------------------------------------------------"
    echo "Finished inference!"
    echo "-------------------------------------------------"
done

# =================================================================================
# =================================================================================

# This script runs a variant of the SoftSeg (Gros et al. 2021) model by training 6 
# different models (on hard labels) for each of the 6 contrasts.


# contrasts=("t1w" "t2w" "t2star" "dwi" "mtoff" "mton")
# cuda_device=2

# for contrast in "${contrasts[@]}"; do

#     echo ""
#     echo "-------------------------------------------------"
#     echo "Running model on contrast(s): ${contrast}"
#     echo "-------------------------------------------------"

#     # run the model
#     CUDA_VISIBLE_DEVICES=${cuda_device} python monai/main.py \
#         -r /home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/datalists/spine-generic/seed15 \
#         -m nnunet -crop 64x192x320 \
#         --contrast ${contrast} \
#         --label-type hard \
#         -initf 32 -me 200 -bs 2 -opt adam -lr 1e-3 -cve 10 -pat 20 \
#         -epb --enable_DS \
#         --loss AdapW
    
#     echo "-------------------------------------------------"
#     echo "Finished training!"
#     echo "-------------------------------------------------"
# done
