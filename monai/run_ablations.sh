#!/bin/bash

# This script runs the ablation study on the different contrasts
# for the contrast-agnostic model with soft segmentation labels.


ablations=("t2w_t1w_dwi_mtoff_t2star" "t2w_t1w_dwi_mtoff" "t2w_t1w_dwi" "t2w_t1w" "t2w")
cuda_device=1

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
