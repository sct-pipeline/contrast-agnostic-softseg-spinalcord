# Towards Contrast-agnostic Soft Segmentation of the Spinal Cord
Official repository for contrast-agnostic spinal cord segmentation project using SoftSeg. 

This repo contains all the code for data preprocessing, training and running inference on other datasets. The code is mainly based on [Spinal Cord Toolbox](https://spinalcordtoolbox.com) and [MONAI](https://github.com/Project-MONAI/MONAI) (PyTorch).

## Table of contents
* [1. Dependencies](#1dependencies)
* [2. Dataset](#2dataset)
* [3. Preprocessing](#3processing)
    * [3.1. Launch processing](#31launch-processing)
    * [3.2. Quality control](#32quality-control)
* [4. Training](#4training)
    * [4.1. Setting up the environment](#41setting-up-the-environment)
    * [4.2. Datalist creation](#42datalist-creation)
    * [4.3. Training](#43training)
    * [4.4. Running inference](#44running-inference)
* [5. Computing morphometric measures (CSA)](#5compute-csa)
    * [5.1. Using MONAI model (best)](#51example-monai)
    * [5.2. Using nnUNet](#52example-nnunet)
* [6. Analyse CSA and QC results](#6analyse-csa-qc-results)
* [7. Get QC reports for other datasets](#7get-qc-reports-for-other-datasets)  
    * [7.1. Example running QC on prediction masks from nnUNet](#71example-running-qc-on-prediction-masks-from-nnunet)
* [8. Active-learning-procedure](#8active-learning-procedure) (TODO)

## 1. Main Dependencies

- [SCT 6.0](https://github.com/neuropoly/spinalcordtoolbox/releases/tag/6.0)
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
Inference can be run on single images using the `monai/run_inference_single_image.py` script. Run `monai/run_inference_single_image.py -h` for usage instructions. Currently, only CPU inference is supported

## 5. Computing morphometric measures (CSA)

To compute CSA at C2-C3 vertebral levels on the prediction masks obtained from the trained models, the script `compute_csa_nnunet.sh` is used. The input is the folder `data_processed_clean` (result from preprocessing) and the path of the prediction masks is added as an extra script argument `-script-args`.
  
For every trained model, you can run:

```
sct_run_batch -jobs -1 -path-data /data_processed_clean/ -path-output <PATH_OUTPUT> -script compute_csa_nnunet.sh -script-args <PATH_PRED_MASKS>
```
* `-path-data`: Path to data from spine generic used for training.
* `-path-output`: Path to save results
* `-script`: Script to compute CSA : `compute_csa_nnunet.sh`
* `-script-args`: Path to the prediction masks

The CSA results will be under `<PATH_OUTPUT>/results`.

### 5.1.Example nnUnet
 **Note:** For nnUnet, change `prefix` in the script `compute_csa_nnunet.sh` according to the preffix in the prediction name.
Here is an example on how to compute CSA on nnUnet models.
```
sct_run_batch -jobs -1 -path-data ~/duke/projects/ivadomed/contrast-agnostic-seg/data_processed_sg_2023-03-10_NO_CROP\data_processed_clean -path-output ~/results -script compute_csa_nnunet.sh -script-args ~/duke/temp/muena/contrast-agnostic/Dataset713_spineGNoCropSoftAvgBin_test
``` 
## 6. Analyse CSA results
To generate violin plots and analyse results, put all CSA results file in the same folder (here `csa_nnunet_vs_ivadomed`) and run:

```
python analyse_csa_all_models.py -i-folder ~/duke/projects/ivadomed/contrast-agnostic-seg/csa_measures_pred/csa_ivadomed_vs_nnunet_vs_monai/ \
                                 -include csa_monai_nnunet_2023-09-18 csa_monai_nnunet_per_contrast csa_gt_2023-08-08 csa_gt_hard_2023-08-08 \
                                          csa_nnunet_2023-08-24 csa_other_methods_2023-09-21-all csa_monai_nnunet_2023-09-18_hard csa_monai_nnunet_diceL
```
* `-i-folder`: Path to folder containing CSA results from models to analyse
* `-include`: names of the folder names to include in the analysis (one model = one foler)

## 7. Run QC report on prediction masks from other datasets

1. Got inside the `scripts` folder:
~~~
cd scripts
~~~
2. Run bash script to generate QC report from prediction masks.
**Note:** For nnUnet, ensure `prefix` in the script `compute_csa_nnunet.sh` according to the preffix in the prediction name and `contrast` with the image contrast.
~~~
sct_run_batch -path-data <PATH_DATA> -path-out <PATH-OUT> -script-args <PATH_PRED_MASK> -jobs 20 -script run_qc_prediction_XXX.sh
~~~
* `-path-data`: Path to the original dataset used to run inferences.
* `-path-output`: Path to the results folder to save QC report
* `-script`: Script `run_qc_prediction_XXX` corresponding to the dataset.
* `-script-args`: Path to prediction masks for the specific dataset
  
### 7.1.Example running QC on prediction masks from nnUnet from other datasets
~~~
sct_run_batch -jobs 20 -path-data ~/data_nvme_sebeda/datasets/dcm-zurich/ \
                       -path-output ~/data_nvme_sebeda/qc_dcm_zurich_sag_nnUnet_2023-05-30 \
                       -script run_qc_prediction_dcm_zurich_sag.sh \
                       -script-args ~/duke/temp/muena/contrast-agnostic/pure-inference/Dataset725_dcmZurichSagittalRPI/test713_softAvg/
~~~

## 8.Active learning procedure
To extend the training set to other contrasts and to pathologies, we applided the segmentation model to other datasets, manually corrected the segmentations and added them to the training set.

Here is the detailed procedure:

1. Run inference on other datasets for the selected models and generate the QC report from prediction masks.
2. Select ~20 interesting images per dataset (using the QC report).
3. Correct the inference on the selected subjects if needed (you can use [`manual-correction`](https://github.com/spinalcordtoolbox/manual-correction) script).
4. Add the inferred segmentations to the `derivatives/labels_contrast_agnostic` folder of each dataset.
5. Add inferred segmentations to the training set (keep the same testing spine generic subjects) & retrain a model.
6. Compute CSA on spine generic testing set and see STD vs before


