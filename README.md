# Towards Contrast-agnostic Soft Segmentation of the Spinal Cord

[![arXiv](https://img.shields.io/badge/arXiv-2310.15402-b31b1b.svg)](https://arxiv.org/abs/2310.15402)

Official repository for contrast-agnostic segmentation of the spinal cord. 

This repo contains all the code for data preprocessing, training and running inference on other datasets. The code for training is based on the [nnUNetv2 framework](https://github.com/MIC-DKFZ/nnUNet). The segmentation model is available as part of [Spinal Cord Toolbox (SCT)](https://spinalcordtoolbox.com) via the `sct_deepseg` functionality.

<details>

<summary> Citation Information </summary>

If you find this work and/or code useful for your research, please cite our paper:

```
@article{BEDARD2025103473,
title = {Towards contrast-agnostic soft segmentation of the spinal cord},
journal = {Medical Image Analysis},
volume = {101},
pages = {103473},
year = {2025},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2025.103473},
url = {https://www.sciencedirect.com/science/article/pii/S1361841525000210},
author = {Sandrine Bédard* and Enamundram Naga Karthik* and Charidimos Tsagkas and Emanuele Pravatà and Cristina Granziera and Andrew Smith and Kenneth Arnold {Weber II} and Julien Cohen-Adad},
note = {Shared authorship -- authors contributed equally}
}
```

</details>

### Updates

#### 2025-02-04

* We have [released](https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases/tag/v3.1) an improved version of the model. The new model has been trained a wide variety of contrast and pathologies and works especially well on compressed spinal cords, MS lesions, etc. 


#### 2025-01-13

* Our paper on proposing a contrast-agnostic soft segmentation of the spinal cord was accepted at Medical Image Analysis! 🎉. Please find the official version of the paper [here](https://www.sciencedirect.com/science/article/pii/S1361841525000210).



## Table of contents
* [1. Using contrast agnostic model with SCT](#1-using-contrast-agnostic-model-with-sct)
* [2. Retraining the contrast-agnostic model ](#2-retraining-the-contrast-agnostic-model)
* [3. Transfer learning from contrast-agnostic model](#3-transfer-learning-from-contrast-agnostic-model)

<!-- * [5. Computing morphometric measures (CSA)](#5-computing-morphometric-measures-csa)
    * [5.1. Using contrast-agnostic model (best)](#51-using-contrast-agnostic-model-best)
    * [5.2. Using nnUNet model](#52-using-nnunet-model)
* [6. Analyse CSA and QC reports](#6-analyse-csa-and-qc-reports)
* [7. Get QC reports for other datasets](#7-get-qc-reports-for-other-datasets)  
    * [7.1. Running QC on predictions from SCI-T2w dataset](#71-running-qc-on-predictions-from-sci-t2w-dataset)
    * [7.2. Running QC on predictions from MS-MP2RAGE dataset](#72-running-qc-on-predictions-from-ms-mp2rage-dataset)
    * [7.3. Running QC on predictions from Radiculopathy-EPI dataset](#73-running-qc-on-predictions-from-radiculopathy-epi-dataset) -->


## 1. Using contrast-agnostic model with SCT

### Installing dependencies

- [Spinal Cord Toolbox (SCT) v7.0](https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/tag/7.0) or higher -- follow the installation instructions [here](https://github.com/spinalcordtoolbox/spinalcordtoolbox?tab=readme-ov-file#installation)
- [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 
- Python (v3.9)

If at the time of installed, SCT v7.0 is not available, download the previous stable version and pull the master branch to get the latest version of the contrast-agnostic model. Once the dependencies are installed, download the latest contrast-agnostic model:

```bash
sct_deepseg -install-task seg_sc_contrast_agnostic
```

### Getting the spinal cord segmentation

To segment a single image, run the following command: 

```bash
sct_deepseg -task seg_sc_lesion_t2w_sci -i <path-to-image>.nii.gz -o <path-to-output-file>.nii.gz 
```

To segment a single image AND generate a QC report for the image, run the following command: 

```bash
sct_deepseg -task seg_sc_lesion_t2w_sci -i <path-to-image>.nii.gz -o <path-to-output-file>.nii.gz -qc qc -qc-subject <name-of-subject>
```

More information on input arguments and their usage can be found [here](https://spinalcordtoolbox.com/stable/user_section/command-line/deepseg/seg_sc_contrast_agnostic.html).

## 2. Retraining the contrast-agnostic model 

The scripts required for re-training the contrast-agnostic model (for reproducibility or extending it to more contrasts/pathologies) can be found in `nnUnet` folder. Training with nnUNet is simple. Once the environment is properly configured, the following steps should get the model up and running:

### Step 1: Configuring the environment

1. Create a conda environment with the following command:
```
conda create -n contrast_agnostic python=3.9
```

2. Activate the environment with the following command:
```
conda activate contrast_agnostic
```

3. Install the required packages with the following command:
```
pip install -r requirements.txt
```

> **Note**
> The `requirements.txt` does not install nnUNet. It has to be installed separately and can be done within the conda environment created above. See [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) for installation instructions. Please note that the nnUNet version used in this work is XXX.


### Step 2: Training the model

The script `train_contrast_agnostic.sh` is a do-it-all script that downloads the datasets from git-annex, creates datalists, converts them into nnUNet-specific format, and trains the model. More instructions about what variables to set and which datasets to use can be found in the script itself. Once these variables are set, the script can be run simply as follows:

```bash
bash train_contrast_agnostic.sh
```
<!-- 
TODO: move to csa_qc_evaluation folder
## 5. Computing morphometric measures (CSA)

To compute the CSA at C2-C3 vertebral levels on the prediction masks and get the QC report of the predictions, the script `compute_csa_qc_<nnunet/monai>.sh` are used. The input is the folder `data_processed_clean` (result from preprocessing) and the path of the prediction masks is added as an extra script argument `-script-args`.
  
For every trained model, you can run:

```
sct_run_batch -jobs -1 -path-data /data_processed_clean/ -path-output <PATH_OUTPUT> -script compute_csa_qc_<nnunet/monai>.sh -script-args <PATH_PRED_MASKS>
```
* `-path-data`: Path to data from spine generic used for training.
* `-path-output`: Path to save results
* `-script`: Script to compute the CSA and QC report
* `-script-args`: Path to the prediction masks

The CSA results will be under `<PATH_OUTPUT>/results` and the QC report under `<PATH_OUTPUT>/qc`.

### 5.1. Using contrast-agnostic model (best)
Here is an example on how to compute CSA and QC on contrast-agnostic model

```
sct_run_batch -jobs -1 -path-data ~/duke/projects/ivadomed/contrast-agnostic-seg/data_processed_sg_2023-03-10_NO_CROP\data_processed_clean -path-output ~/results -script compute_csa_qc_monai.sh -script-args ~/duke/projects/ivadomed/contrast-agnostic-seg/models/monai/spine-generic-results
```

### 5.2. Using nnUNet model
 **Note:** For nnUnet, change the variable `prefix` in the script `compute_csa_nnunet.sh` according to the prefix in the prediction name.
Here is an example on how to compute CSA and QC on nnUNet models.

```
sct_run_batch -jobs -1 -path-data ~/duke/projects/ivadomed/contrast-agnostic-seg/data_processed_sg_2023-03-10_NO_CROP\data_processed_clean -path-output ~/results -script compute_csa_qc_nnunet.sh -script-args ~/duke/projects/ivadomed/contrast-agnostic-seg/models/nnunet/spine-generic-results/test_predictions_2023-08-24
``` 

TODO: Move to csa_generate_figures folder
## 6. Analyse CSA and QC reports
To generate violin plots and analyse results, put all CSA results file in the same folder (here `csa_ivadomed_vs_nnunet_vs_monai`) and run:

```
python analyse_csa_all_models.py -i-folder ~/duke/projects/ivadomed/contrast-agnostic-seg/csa_measures_pred/csa_ivadomed_vs_nnunet_vs_monai/ \
                                 -include csa_monai_nnunet_2023-09-18 csa_monai_nnunet_per_contrast csa_gt_2023-08-08 csa_gt_hard_2023-08-08 \
                                          csa_nnunet_2023-08-24 csa_other_methods_2023-09-21-all csa_monai_nnunet_2023-09-18_hard csa_monai_nnunet_diceL
```
* `-i-folder`: Path to folder containing CSA results from models to analyse
* `-include`: names of the folder names to include in the analysis (one model = one folder)

The plots will be saved to the parent directory with the name `charts_<datetime.now())>` -->


## 3. Transfer learning from contrast-agnostic model

This section provides instructions on how to train nnUNet models with pre-trained weights of the contrast-agnostic model. Ideal use cases include using the 
pretrained weights for training/finetuning on any spinal-cord-related segmentation task (e.g. lesions, rootlets, etc.). 

### Step 1: Download the pretrained weights

Download the pretrained weights from the [latest release](https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases) of the 
contrast-agnostic model. This refers to the `.zip` file of the format `model_contrast_agnostic_<date>_nnunet_compatible.zip`. 

> [!WARNING]  
> Only download the model with the `nnunet_compatible` suffix. If a release does not have this suffix, then that model weights are not directly 
compatible with nnUNet.


### Step 2: Create a new plans file

1. After running `nnUNetv2_plan_and_preprocess` (and before running `nnUNetv2_predict`), create a copy of the original `nnUNetPlans.json` file (found under `$nnUNet_preprocessed/<dataset_name_or_id>`) and  rename it to `nnUNetPlans_contrast_agnostic.json`. 
2. In the `nnUNetPlans_contrast_agnostic.json`, modify the values of the following keys in the `3d_fullres` dict to be able to match the 
values used for the contrast-agnostic model:

```json

"patch_size": [64, 192, 320],
"n_stages": 6,
"features_per_stage": [32, 64, 128, 256, 320, 320],
"n_conv_per_stage": [2, 2, 2, 2, 2, 2],
"n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
"strides": [
    [1, 1, 1],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [1, 2, 2]
    ],
"kernel_sizes":[
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3]
    ],

```

> [!IMPORTANT]  
> * Note that the patch size does not have to be `[64, 192, 320]`. Any patch size can be used as long as the RL dimension is divisible by 16, AP and SI dimensions are divisible by 32. Examples of other patch sizes: `[16, 32, 32]`, `[32, 64, 96]`, `[64, 128, 64]`, `[64, 160, 192]`, `[64, 192, 256]` etc. For possibly best results, use the patch sizes that are close to original patch size in `nnUNetPlans.json`.

### Step 3: Train/Finetune the nnUNet model on your task

Provide the path to the downloaded pretrained weights:

```bash

nnUNetv2_train <dataset_name_or_id> <configuration> <fold> -p nnUNetPlans_contrast_agnostic -pretrained_weights <path_to_pretrained_weights> -tr <nnUNetTrainer_Xepochs>

```

> [!IMPORTANT]  
> * Training/finetuning with contrast-agnostic weights only works when for 3D nnUNet models. 
> * Ensure that all images are in the RPI orientation before running `nnUNetv2_plan_and_preprocess`. This is because the updated 
`patch_size` refers to patches in RPI orientation (if images are in different orientation, then the patch size might be sub-optimal).
> * Ensure that `X` in `nnUNetTrainer_Xepochs` is set to a lower value than 1000. The idea is the finetuning does not require as 
many epochs as training from scratch because the contrast-agnostic model has already been trained on a lot of spinal cord images 
(so it might not require 1000 epochs to converge).
> * The modified `nnUNetPlans_contrast_agnostic.json` might not have the same values for parameters such as `patch_size`, `strides`, 
`n_stages`, etc. automatically set by nnUNet. As a result, the original (train-from-scratch) model might even perform better. 
In such cases, training/finetuning with contrast-agnostic weights should just be considered as another baseline for your actual task.



