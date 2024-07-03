#!/bin/bash

contrast_ablations=(
    "t2w_t1w_dwi_mtoff_t2star" 
    "t2w_t1w_dwi_mtoff" 
    "t2w_t1w_dwi" 
    "t2w_t1w" 
    "t2w"
)

models=(
    "nnunet_t2w_t1w_dwi_mtoff_t2star_soft_nf=32_opt=adam_lr=0.001_AdapW_bs=2_64x192x320_20240617-0850" 
    "nnunet_t2w_t1w_dwi_mtoff_soft_nf=32_opt=adam_lr=0.001_AdapW_bs=2_64x192x320_20240618-0920" 
    "nnunet_t2w_t1w_dwi_soft_nf=32_opt=adam_lr=0.001_AdapW_bs=2_64x192x320_20240619-0952" 
    "nnunet_t2w_t1w_soft_nf=32_opt=adam_lr=0.001_AdapW_bs=2_64x192x320_20240620-1319" 
    "nnunet_t2w_soft_nf=32_opt=adam_lr=0.001_AdapW_bs=2_64x192x320_20240620-2349" 
)


# run a for loop upto 5
for i in {0..4}; do 
    contrast=${contrast_ablations[$i]}
    model=${models[$i]}
    
    echo "---------------------------------------------------------------"
    echo "Running sct_run_batch for: contrast:"
    echo "\t- contrast: $contrast"
    echo "\t- model: $model"
    echo "---------------------------------------------------------------"

    # run batch
    sct_run_batch \
        -path-data ~/duke/projects/ivadomed/contrast-agnostic-seg/data_processed_sg_2023-08-08_NO_CROP/data_processed_clean \
        -script csa_qc_evaluation_spine_generic/compute_csa_qc_monai.sh \
        -path-output ~/contrast-agnostic/r1_revision_exps/results_csa/csa_ablation_monai_soft_${contrast}_20240625 \
        -script-args ~/contrast-agnostic/r1_revision_exps/results/${model} \
        -jobs 8

done

# # run batch
# sct_run_batch \
#     -path-data ~/duke/projects/ivadomed/contrast-agnostic-seg/data_processed_sg_2023-08-08_NO_CROP/data_processed_clean \
#     -script csa_qc_evaluation_spine_generic/compute_csa_qc_monai.sh \
#     -path-output ~/contrast-agnostic/r1_revision_exps/results_csa/csa_monai_hard_diceCE_20240625 \
#     -script-args ~/contrast-agnostic/r1_revision_exps/results/nnunet_all_hard_nf=32_opt=adam_lr=0.001_DiceCE_bs=2_64x192x320_20240613-1630 \
#     -jobs 8 