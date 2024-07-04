#!/bin/bash

# # This script runs the ablation study on the different contrasts
# # for the contrast-agnostic model with soft segmentation labels.


# ablations=("t2w_t1w_dwi_mtoff_t2star" "t2w_t1w_dwi_mtoff" "t2w_t1w_dwi" "t2w_t1w" "t2w")
# cuda_device=1

# for ablation in "${ablations[@]}"; do

#     echo ""
#     echo "-------------------------------------------------"
#     echo "Running ablation on contrast(s): ${ablation}"
#     echo "-------------------------------------------------"

#     # run the model
#     CUDA_VISIBLE_DEVICES=${cuda_device} python monai/main.py \
#         -r /home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/datalists/spine-generic-contrast-ablations/seed15 \
#         -m nnunet -crop 64x192x320 \
#         --contrast ${ablation} \
#         --label-type soft \
#         -initf 32 -me 200 -bs 2 -opt adam -lr 1e-3 -cve 5 -pat 20 \
#         -epb --enable_DS \
#         --loss AdapW
    
#     echo "-------------------------------------------------"
#     echo "Finished training!"
#     echo "-------------------------------------------------"
# done


# =================================================================================
# =================================================================================

# This script runs a variant of the SoftSeg (Gros et al. 2021) model by training 6 
# different models (on hard labels) for each of the 6 contrasts.


contrasts=("t1w" "t2w" "t2star" "dwi" "mtoff" "mton")
cuda_device=2

for contrast in "${contrasts[@]}"; do

    echo ""
    echo "-------------------------------------------------"
    echo "Running model on contrast(s): ${contrast}"
    echo "-------------------------------------------------"

    # run the model
    CUDA_VISIBLE_DEVICES=${cuda_device} python monai/main.py \
        -r /home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/datalists/spine-generic/seed15 \
        -m nnunet -crop 64x192x320 \
        --contrast ${contrast} \
        --label-type hard \
        -initf 32 -me 200 -bs 2 -opt adam -lr 1e-3 -cve 10 -pat 20 \
        -epb --enable_DS \
        --loss AdapW
    
    echo "-------------------------------------------------"
    echo "Finished training!"
    echo "-------------------------------------------------"
done
