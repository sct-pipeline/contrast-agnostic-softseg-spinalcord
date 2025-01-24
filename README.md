# Towards Contrast-agnostic Soft Segmentation of the Spinal Cord

[![arXiv](https://img.shields.io/badge/arXiv-2310.15402-b31b1b.svg)](https://arxiv.org/abs/2310.15402)

Official repository for contrast-agnostic spinal cord segmentation project using SoftSeg. 

This repo contains all the code for data preprocessing, training and running inference on other datasets. The code is mainly based on [Spinal Cord Toolbox](https://spinalcordtoolbox.com) and [MONAI](https://github.com/Project-MONAI/MONAI) (PyTorch).

**CITATION INFO**: If you find this work and/or code useful for your research, please cite our paper:

```
@article{bedard2023towards,
  title={Towards contrast-agnostic soft segmentation of the spinal cord},
  author={B{\'e}dard, Sandrine and Enamundram, Naga Karthik and Tsagkas, Charidimos and Pravat{\`a}, Emanuele and Granziera, Cristina and Smith, Andrew and Weber II, Kenneth Arnold and Cohen-Adad, Julien},
  journal={arXiv preprint arXiv:2310.15402},
  year={2023}
  url={https://arxiv.org/abs/2310.15402}
}
```

## Table of contents
* [1. Main Dependencies](#1-main-dependencies)
* [2. Dataset](#2-dataset)
* [3. Preprocessing](#3-preprocessing)
    * [3.1. Launch preprocessing](#31-launch-preprocessing)
    * [3.2. Quality control](#32-quality-control)
* [4. Training](#4-training)
    * [4.1. Setting up the environment](#41-setting-up-the-environment)
    * [4.2. Datalist creation](#42-datalist-creation)
    * [4.3. Training](#43-training)
    * [4.4. Running inference](#44-running-inference)
    * [4.5. Compute ANIMA metrics](#45-compute-anima-metrics)
* [5. Computing morphometric measures (CSA)](#5-computing-morphometric-measures-csa)
    * [5.1. Using contrast-agnostic model (best)](#51-using-contrast-agnostic-model-best)
    * [5.2. Using nnUNet model](#52-using-nnunet-model)
* [6. Analyse CSA and QC reports](#6-analyse-csa-and-qc-reports)
* [7. Get QC reports for other datasets](#7-get-qc-reports-for-other-datasets)  
    * [7.1. Running QC on predictions from SCI-T2w dataset](#71-running-qc-on-predictions-from-sci-t2w-dataset)
    * [7.2. Running QC on predictions from MS-MP2RAGE dataset](#72-running-qc-on-predictions-from-ms-mp2rage-dataset)
    * [7.3. Running QC on predictions from Radiculopathy-EPI dataset](#73-running-qc-on-predictions-from-radiculopathy-epi-dataset)
* [8. Using contrast-agnostic model with nnUNet](#8-use-contrast-agnostic-model-with-nnunet)

## 1. Main Dependencies

- [SCT 6.2](https://github.com/neuropoly/spinalcordtoolbox/releases/tag/6.2)
- Python 3.9

## 2. Dataset
The source data can be found at [spine-generic multi-subject](https://github.com/spine-generic/data-multi-subject/).

The preprocessed data are located at `duke:projects/ivadomed/contrast-agnostic-seg/data_processed_sg_2023-08-08_NO_CROP\data_processed_clean` (internal server)

## 3. Preprocessing
Main preprocessing steps include:

For T1w and T2w:
* Resampling
* Reorientation to RPI
* Remove suffix `_RPI_r`

For T2star:
* Compute root-mean square across 4th dimension (if it exists)
* Remove suffix `_rms`

For DWI:
* Generate mean image after motion correction
* Remove intermediate files

Next steps are to generate the contrast-agnostic soft segmentation: 
- creates spinal cord segmentation (if it doesn't exist yet)
- creates a mask around the T2 spinal cord
- co-register all contrasts to the T2 spinal cord 
- average all segmentations from each contrast within the same space (the T2)
- bring back the segmentations to the original image space of each contrast (except for the T2)

The output of this script is a new `derivatives/labels_softseg/` folder that contains the soft labels to be used in this contrast-agnostic segmentation project. All the registration were manually QC-ed (see [Quality Control](#quality-control)) and the problematic registrations were listed in [exclude.yml](https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/blob/main/exclude.yml). The processing was run again to generate the soft segmentations. 

Specify the path of preprocessed dataset with the flag `-path-data`. 

### 3.1. Launch Preprocessing

This section assumes that SCT is installed. The installation instructions can be found [here](https://spinalcordtoolbox.com/en/latest/user_section/installation.html).

```
cd processing_spine_generic
sct_run_batch -jobs -1 -path-data <PATH_DATA> -path-output <PATH-OUTPUT> -script process_data.sh -script-args exclude.yml
```

or use a config file:

```
Example config_process_data.json: 
{
  "path_data"   : "~/data_nvme_sebeda/datasets/data-multi-subject/",
  "path_output" : "~/data_nvme_sebeda/data_processed_sg_2023-08-04_NO_CROP",
  "script"      : "process_data.sh",
  "jobs"        : 50,
  "exclude_list": ["sub-brnoUhb02", "sub-brnoUhb03", "sub-brnoUhb07", "sub-brnoUhb08", "sub-brnoUhb08", "sub-brnoUhb08", "sub-ucdavis01", "sub-ucdavis02", "sub-ucdavis03", "sub-ucdavis04", "sub-ucdavis05", "sub-ucdavis06", "sub-ucdavis07", "sub-beijingVerio01", "sub-beijingVerio02", "sub-beijingVerio03", "sub-beijingVerio04", "sub-beijingGE01", "sub-beijingGE02", "sub-beijingGE03", "sub-beijingGE04", "sub-ubc01", "sub-oxfordOhba02"]
}
```
```
sct_run_batch -config config_process_data.json
```

A `process_data_clean` folder is created in <PATH-OUTPUT> where the cropped data and derivatives are included. Here, only the images that have a manual segmentation and soft segmentation are transfered.

### 3.2. Quality control

After running the analysis, check your Quality Control (QC) report by opening the file <PATH-OUTPUT>/qc/index.html. Use the "search" feature of the QC report to quickly jump to segmentations or labeling issues.

**1. Segmentations**

If segmentation issues are noticed while checking the quality report, proceed to manual correction using the procedure below:

* In QC report, search for "deepseg" to only display results of spinal cord segmentation
* Review segmentation and spinal cord.
* Click on the F key to indicate if the segmentation is OK ✅, needs manual correction ❌ or if the data is not usable ⚠️ (artifact). Two .yml lists, one for manual corrections and one for the unusable data, will automatically be generated.
* Download the lists by clicking on **Download QC Fails*** and on **Download QC Artifacts**.

Proceed to manual correction using FSLeyes or ITK snap. Upload the manual segmentations (_seg-manual.nii.gz) with json sidecar in the derivatives.
Re-run the analysis: [Launch processing](#launch-processing)

**2. Registrations**

* In QC report, search for "sct_register_multimodal" to only display results of registration.
* Click on the F key to indicate if the registration is OK ✅, needs to be excluded ❌ or if the data is not usable ⚠️ (artifact). Two .yml lists, will automatically be generated.
* Download the list by clicking on **Download QC Fails** and add the file names under `FILES_REG` to https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/blob/main/exclude.yml
* Re-run the analysis: [Launch processing](#launch-processing)

**3. Soft segmentations**

* In QC report, search for "softseg" to only display results of spinal cord soft segmentation
* Click on the F key to indicate if the soft segmentation is OK ✅, needs manual correction ❌ or if the data is not usable ⚠️ (artifact). Two .yml lists, one for manual corrections and one for the unusable data, will automatically be generated.
* Download the list by clicking on **Download QC Fails**.
* Upload the soft segmentations under `data-multi-subject/derivatives/labels_softseg`.
* Re-run the analysis: [Launch processing](#launch-processing)


## 4. Training

### 4.1. Setting up the environment

The following commands show how to set up the environment. Note that the documentation assumes that the user has `conda` installed on their system. Instructions on installing `conda` can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. Create a conda environment with the following command:
```
conda create -n venv_monai python=3.9
```

2. Activate the environment with the following command:
```
conda activate ven_monai
```

3. Install the required packages with the following command:
```
pip install -r monai/requirements.txt
```

### 4.2. Datalist creation

The training script expects a datalist file in the Medical Decathlon format containing image-label pairs. The datalist can be created by running the `create_msd_data.py` script. For example, creating the datalist for the `soft_all` model:
```
python monai/create_msd_data.py -pd ~/duke/projects/ivadomed/contrast-agnostic-seg/data_processed_sg_2023-08-08_NO_CROP\data_processed_clean> -po ~/datasets/contrast-agnostic/ --contrast all --label-type soft --seed 42
```

The dataset split containing the training, validation, and test subjects can be found in the `monai/data_split_all_soft_seed15.yaml` file. 

> **Note** 
> The output of the above command is just `.json` file pointing to the image-label pairs in the original BIDS dataset. It _does not_ copy the existing data to the output folder. 

### 4.3. Training
The training uses MONAI functions and is written in PyTorch Lightning. Example training command to run the `soft_all` model:
```
python monai/main.py -m nnunet -crop 64x192x320 --contrast all --label-type soft -initf 32 -me 200 -bs 2 -opt adam -lr 1e-3 -cve 5 -pat 20 -epb -stp --enable_DS
```

Example training command to run the `soft_per_contrast` model on the `dwi` contrast:
```
python monai/main.py -m nnunet -crop 64x192x320 --contrast dwi --label-type soft -initf 32 -me 3 -bs 2 -opt adam -lr 1e-3 -cve 5 -pat 20 -epb -stp --enable_DS
```

These commands assume that the datalist created in [Section 4.2](#42datalist-creation#) lies in the same folder as `monai/main.py`. Run `python monai/main.py -h` to see all the available arguments and their descriptions.

> **Note**
> WandB is used experiment tracking and is implemented via Lightning's Wandblogger. Make sure that the `project` and `entity` are changed to the appropriate values.

### 4.4. Running inference
Inference can be run on single images using the `monai/run_inference_single_image.py` script. Run `monai/run_inference_single_image.py -h` for usage instructions. Both CPU and GPU-based inference are supported.

### 4.5. Compute ANIMA metrics
To compute the ANIMA metrics shown in the paper, the scripts `compute_anima_metrics_*.py` are used. For generating the metrics on the spine-generic dataset and also for deepseg and propseg methods, use the following command:

```
python anima_metrics/compute_anima_metrics_spine_generic.py --pred-folder <PATH_PREDS> --method <monai/deepseg2d/deepseg3d/propseg> -dname spine-generic
```

For reproducing the results on the other datasets, use the following command:
```
python anima_metrics/compute_anima_metrics_unseen_datasets.py --pred-folder <PATH_PREDS> -dname <sci-t2w/ms-mp2rage/radiculopathy-epi>
```

> **Note**
> The `--pred-folder` argument expects the path to the folder containing the prediction and GT segmentation masks. 


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

## 6. Analyse CSA and QC reports
To generate violin plots and analyse results, put all CSA results file in the same folder (here `csa_ivadomed_vs_nnunet_vs_monai`) and run:

```
python analyse_csa_all_models.py -i-folder ~/duke/projects/ivadomed/contrast-agnostic-seg/csa_measures_pred/csa_ivadomed_vs_nnunet_vs_monai/ \
                                 -include csa_monai_nnunet_2023-09-18 csa_monai_nnunet_per_contrast csa_gt_2023-08-08 csa_gt_hard_2023-08-08 \
                                          csa_nnunet_2023-08-24 csa_other_methods_2023-09-21-all csa_monai_nnunet_2023-09-18_hard csa_monai_nnunet_diceL
```
* `-i-folder`: Path to folder containing CSA results from models to analyse
* `-include`: names of the folder names to include in the analysis (one model = one folder)

The plots will be saved to the parent directory with the name `charts_<datetime.now())>`


## 7. Get QC reports for other datasets

The QC reports from three other datasets `sci-t2w`, `ms-mp2rage`, and `radiculopathy-epi` are shown in the paper. The scripts for reproducing the results are in the `qc_other_datasets` folder.

General command to run QC on prediction masks from other datasets:
~~~
sct_run_batch -path-data <PATH_DATA> -path-out <PATH-OUT> -script-args <PATH_PRED_MASK> -jobs 20 -script run_qc_prediction_<dataset>.sh
~~~
* `-path-data`: Path to the original dataset used to run inferences.
* `-path-output`: Path to the results folder to save QC report
* `-script`: Script `run_qc_prediction_<dataset>` corresponding to the dataset.
* `-script-args`: Path to prediction masks for the specific dataset
  
### 7.1. Running QC on predictions from SCI-T2w dataset

Using the contrast-agnostic model:
~~~
sct_run_batch -jobs 32 -path-data ~/path-to-dataset/sci-colorado/ \
                       -path-output ~/path-to-output/qc_contrast-agnostic_sci-colorado \
                       -script run_qc_prediction_sci_colorado.sh \
                       -script-args ~/duke/projects/ivadomed/contrast-agnostic-seg/models/monai/sci-colorado-results/test_preds_colorado_soft_all
~~~

Using the nnUNet model:
~~~
sct_run_batch -jobs 32 -path-data ~/path-to-dataset/sci-colorado/ \
                       -path-output ~/path-to-output/qc_nnunet_sci-colorado \
                       -script run_qc_prediction_sci_colorado.sh \
                       -script-args "/home/GRAMES.POLYMTL.CA/u114716/duke/projects/ivadomed/contrast-agnostic-seg/models/nnunet/sci-colorado-results/test_predictions nnUNet"
~~~

### 7.2. Running QC on predictions from MS-MP2RAGE dataset

Using the contrast-agnostic model:
~~~
sct_run_batch -jobs 32 -path-data ~/path-to-dataset/basel-mp2rage/ \
                       -path-output ~/path-to-output/qc_contrast-agnostic_basel-mp2rage \
                       -script run_qc_prediction_basel_mp2rage.sh \
                       -script-args ~/duke/projects/ivadomed/contrast-agnostic-seg/models/monai/basel-mp2rage-rpi-results/test_preds_mp2rage_soft_all
~~~

Using the nnUNet model:
~~~
sct_run_batch -jobs 32 -path-data ~/path-to-dataset/basel-mp2rage/ \
                       -path-output ~/path-to-output/qc_nnunet_basel-mp2rage \
                       -script run_qc_prediction_basel_mp2rage.sh \
                       -script-args "/home/GRAMES.POLYMTL.CA/u114716/duke/projects/ivadomed/contrast-agnostic-seg/models/nnunet/basel-mp2rage-rpi-results/test_predictions nnUNet"
~~~

### 7.3. Running QC on predictions from Radiculopathy-EPI dataset

Using the contrast-agnostic model:
~~~
sct_run_batch -jobs 32 -path-data ~/path-to-dataset/epi-stanford/ \
                       -path-output ~/path-to-output/qc_contrast-agnostic_epi-stanford \
                       -script run_qc_prediction_epi_stanford.sh \
                       -script-args ~/duke/projects/ivadomed/contrast-agnostic-seg/models/monai/epi-stanford-results/test_preds_soft_all
~~~

Using the nnUNet model:
~~~
sct_run_batch -jobs 32 -path-data ~/path-to-dataset/epi-stanford/ \
                       -path-output ~/path-to-output/qc_nnunet_radiculopathy-epi \
                       -script run_qc_prediction_epi_stanford.sh \
                       -script-args "/home/GRAMES.POLYMTL.CA/u114716/duke/projects/ivadomed/contrast-agnostic-seg/models/nnunet/epi-stanford-results/test_preds_stanford_rest_weber nnUNet"
~~~


## 8. Transfer learning from contrast-agnostic model

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



