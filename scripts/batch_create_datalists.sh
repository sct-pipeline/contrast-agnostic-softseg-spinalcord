#!/bin/bash

# Unified script for creating datalist jsons

# seed
SEED=50

# script path
PATH_SCRIPT="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/contrast-agnostic-softseg-spinalcord"

# list of datasets to train on
# NOTE: not using nih-ms-mp2rage (bad GT)
DATASETS=("data-multi-subject" "basel-mp2rage" "canproco" \
            "lumbar-epfl" "lumbar-vanderbilt" "dcm-brno" "dcm-zurich" "dcm-zurich-lesions" "dcm-zurich-lesions-20231115" \
            "sci-paris" "sci-zurich" "sci-colorado" \
            "data-praxis-sci/site_006" "data-praxis-sci/site_007"
            )

# base root path for the datasets
PATH_DATA_BASE="/home/GRAMES.POLYMTL.CA/u114716/datasets"

# output path
folder_name=$(date +"%Y%m%d")-v21PtrAll
PATH_OUTPUT="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/datalists/${folder_name}"

# create datalists!

for dataset in ${DATASETS[@]}; do
    
    echo "-----------------------------------"
    echo "Creating datalist for ${dataset}"
    echo "-----------------------------------"
    
    python ${PATH_SCRIPT}/monai/create_msd_data.py \
        --seed ${SEED} \
        --path-data ${PATH_DATA_BASE}/${dataset} \
        --path-out ${PATH_OUTPUT}
done

echo "-----------------------------------"
echo "Done!"
echo "Datalists created in ${PATH_OUTPUT}"
echo "-----------------------------------"