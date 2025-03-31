# Towards Contrast-agnostic Soft Segmentation of the Spinal Cord

[![arXiv](https://img.shields.io/badge/arXiv-2310.15402-b31b1b.svg)](https://arxiv.org/abs/2310.15402)

Official repository for contrast-agnostic segmentation of the spinal cord. 

This repo contains all the code for data preprocessing, training and running inference on other datasets. The code for training is based on the [nnUNetv2 framework](https://github.com/MIC-DKFZ/nnUNet). The segmentation model is available as part of [Spinal Cord Toolbox (SCT)](https://spinalcordtoolbox.com/stable/user_section/command-line/deepseg/seg_sc_contrast_agnostic.html) via the `sct_deepseg` functionality.


### Citation Information

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

**TODO**: add lifelong learning figure


## Table of contents
2. [Training the model ](#2-training-the-model)
3. [Lifelong learning for monitoring morphometric drift](#3-lifelong-learning-for-monitoring-morphometric-drift)

<!-- * [5. Computing morphometric measures (CSA)](#5-computing-morphometric-measures-csa)
    * [5.1. Using contrast-agnostic model (best)](#51-using-contrast-agnostic-model-best)
    * [5.2. Using nnUNet model](#52-using-nnunet-model)
* [6. Analyse CSA and QC reports](#6-analyse-csa-and-qc-reports)
* [7. Get QC reports for other datasets](#7-get-qc-reports-for-other-datasets)  
    * [7.1. Running QC on predictions from SCI-T2w dataset](#71-running-qc-on-predictions-from-sci-t2w-dataset)
    * [7.2. Running QC on predictions from MS-MP2RAGE dataset](#72-running-qc-on-predictions-from-ms-mp2rage-dataset)
    * [7.3. Running QC on predictions from Radiculopathy-EPI dataset](#73-running-qc-on-predictions-from-radiculopathy-epi-dataset) -->


## 2. Training the model 

The scripts required for training the model can be found in `nnUnet` folder. Training with nnUNet is simple. Once the environment is properly configured, the following steps should get the model up and running:

### Step 1: Configuring the environment

1. Create a conda environment with the following command:
```bash
conda create -n contrast_agnostic python=3.9
```

2. Activate the environment with the following command:
```bash
conda activate contrast_agnostic
```

3. Clone the repository with the following command:
```bash
git clone https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord.git
```

3. Install the required packages with the following command:
```bash
cd contrast-agnostic-softseg-spinalcord/nnUnet
pip install -r requirements.txt
```

> **Note**
> The `requirements.txt` does not install nnUNet. It has to be installed separately and can be done within the conda environment created above. See [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) for installation instructions. Please note that the nnUNet version used in this work is tag `v2.5.1`.


### Step 2: Training the model

The script `scripts/train_contrast_agnostic.sh` is a do-it-all script that downloads the datasets from git-annex, creates datalists, converts them into nnUNet-specific format, and trains the model. More instructions about what variables to set and which datasets to use can be found in the script itself. Once these variables are set, the script can be run simply as follows:

```bash
bash scripts/train_contrast_agnostic.sh
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


## 3. Lifelong learning for monitoring morphometric drift

This section provides some notes on the lifelong/continuous learning framework for automatically monitoring morphometric drift between various versions of segmentation models. Once a new segmentation model is developed and released, a GitHub actions (GHA) workflow is triggered which automatically computes the spinal cord CSA between current (new) version of the model and previously released models. 

For a fair comparison, we evalute various model versions on the frozen test set of the spine-generic `data-multi-subject` (public) dataset. The test split can be found in `scripts/spine_generic_test_split_for_csa_drift_monitoring.yaml` file. 

### Step 1: Creating a new release

Here are the steps involved in the workflow:

* After training a new segmentation model, create a release with the following naming convention: 
    * **Tag name**: `vX.Y` (e.g. `v2.0`, `v3.0`, etc.), where `X` is the major update (i.e. architectural/training-strategy change) and `Y` is the minor update (addition of new contrasts and/or pathologies).
    * **Release title**: `contrast-agnostic-spinal-cord-segmentation vX.Y` (note, the title can be anything, GHA workflow does not depend on it).
    * **Release description**: A drop-down summary of the dataset characteristics. The details of the datasets used during training is automatically generated from the `nnUnet/utils.py` script.
    * **Release assets**: The model weights and the training logs (if needed) are attached to the release. The entire output folder of the nnUnet model containing the folds, should be uploaded. The naming convention for the `.zip` file should be `model_contrast_agnostic_<date-the-model-was-trained-on>.zip`. 
    * Once the above steps are completed, publish the release.

### Step 2: The GHA workflow

* Once published, the release triggers a GHA workflow. The workflow is a `.yml` file located in the `.github/workflows` folder. For a high-level overview, it is divided into the following steps:
    * **Job 1**: Clones the dataset via git-annex and only downloads subjects in the test split. The dataset is cached for future use.
    * **Job 2**: The test set of *(n=49)* is split into batches of 3 subjects for parallel processing. The model is downloaded from the release and each job (or, a runner) is responsible for computing the C2-C3 CSA for all the 6 contrasts. 
    * **Job 3**: The output `.csv` files are aggregated across batches and merged into a single CSV file. The file is saved with the following naming convention `csa_c2c3__model_<tag-name>.csv` (note that the tag name defined in Step 1 is being used here) and uploaded to the release.
    * **Job 4**: All `csa_c2c3__model_<tag-name>.csv` files corresponding to current and previous releases are downloaded. Then, violin plots comparing the CSA per contrast (for each model) and the STD of CSA across contrasts are generated. The plots are saved in the `morphometric_plots.zip` folder and uploaded to the existing release.

In summary, once a new model is released, the GitHub actions workflow automatically generates the plots for monitoring the morphometric drift between various versions of the segmentation model.


### Updates

#### 2025-02-04

* We have [released](https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases/tag/v3.1) an improved version of the model. The new model has been trained a wide variety of contrast and pathologies and works especially well on compressed spinal cords, MS lesions, etc. 


#### 2025-01-13

* Our paper on proposing a contrast-agnostic soft segmentation of the spinal cord was accepted at Medical Image Analysis! 🎉. Please find the official version of the paper [here](https://www.sciencedirect.com/science/article/pii/S1361841525000210).
