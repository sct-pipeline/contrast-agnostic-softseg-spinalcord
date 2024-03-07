#!/bin/bash
# 
# This script does the following:
# 1. Runs inference on the test set using the trained monai models across different label types
# 2. Generates CSA violin plots 
# 
# Usage:
#   bash analyze_results_across_labels.sh <path_to_sct_config>
# 
# Example config file:
# {
#     "path_data"   : "<path_to_processed_spine-generic_dataset>",
#     "path_output" : "<path_to_output_directory>",
#     "script"      : "csa_qc_evaluation_spine_generic/comparison_across_training_labels.py",
#     "jobs"        : 5,
#     "script_args" : "nnUnet/run_inference_single_subject.py <path-to-trained-nnUnet-model> monai/run_inference_single_image.py <path-to-monai-model-label-type-1> <path-to-monai-model-label-type-2>",
#     "include_list": ["sub-amu02", "sub-amu05", "sub-balgrist06", "sub-barcelona05", "sub-beijingPrisma02", "sub-beijingPrisma03", "sub-brnoUhb04", "sub-brnoUhb06", "sub-cardiff01", "sub-cmrra01", "sub-cmrra04", "sub-cmrrb01", "sub-cmrrb07", "sub-fslPrisma05", "sub-geneva03", "sub-hamburg03", "sub-mgh02", "sub-mgh05", "sub-milan02", "sub-milan04", "sub-milan05", "sub-mniS06", "sub-mountSinai01", "sub-nottwil04", "sub-oxfordFmrib02", "sub-oxfordOhba03", "sub-oxfordOhba04", "sub-oxfordOhba05", "sub-pavia02", "sub-pavia03", "sub-pavia05", "sub-perform05", "sub-sherbrooke02", "sub-stanford06", "sub-strasbourg03", "sub-strasbourg05", "sub-tehranS02", "sub-tehranS04", "sub-tehranS05", "sub-tokyo750w03", "sub-tokyo750w05", "sub-tokyo750w07", "sub-tokyoSkyra07", "sub-ucl03", "sub-unf05", "sub-vuiisAchieva01", "sub-vuiisAchieva02", "sub-vuiisAchieva06", "sub-vuiisIngenia04"]
# }
   

# Uncomment for full verbose
# set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Set the following variables to the desired values
PATH_SCT_CONFIG=$1

# Run the sct_run_batch script
sct_run_batch -config $PATH_SCT_CONFIG

# get "path_output" key from the config file
PATH_OUTPUT=$(python -c "import json; print(json.load(open('$PATH_SCT_CONFIG'))['path_output'])")

# Generate CSA violin plots
python csa_generate_figures/analyse_csa_across_training_labels.py \
    -i ${PATH_OUTPUT}/results/csa_label_types_c24.csv