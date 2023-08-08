# Contrast-agnostic soft segmentation of the spinal cord
Contrast-agnostic spinal cord segmentation project with softseg

This repo creates a series of preparations for comparing the newly trained ivadomed models (Pytorch based), with the old models that are currently implemented in spinal cord toolbox [SCT] (tensorflow based).

## Dependencies

- [SCT 5.3.0](https://github.com/neuropoly/spinalcordtoolbox/releases/tag/5.3.0)
- Python 3.7.

## Dataset
The source data are from the [spine-generic multi-subject](https://github.com/spine-generic/data-multi-subject/).

The processed data are located on `duke:projects/ivadomed/contrast-agnostic-seg/data`.

> ⚠️ Currently, there are three processed datasets to account for the issue of ivadomed loader that cannot deal with the MTS files (https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/issues/25). In the future, there should be only ONE processed dataset. 

## Processing
Main processing steps include:

For T1w and T2w :
* Resampling
* Reorient to RPI
* Remove suffix `_RPI_r`

For T2star:
* Compute root-mean square across 4th dimension (if it exists)
* Remove suffix `_rms`

For DWI:
* Generate mean image after motion correction
* Remove intermidiate files

Next steps are to generate a contrast-agnostic soft segmentation: 
- creates spinal cord segmentation (if it doesn't exist yet)
- creates a mask around the T2 spinal cord
- co-register all contrasts to the T2 spinal cord 
- average all segmentations from each contrast within the same space (the T2)
- bring back the segmentations to the original image space of each contrast (except for the T2)
- Crop images, segmentations and soft segmentation around the spinal cord

The output of this script is a new `derivatives/labels_softseg/` folder that contains the soft labels to be used in this contrast-agnostic segmentation project. All the registration were manually QC-ed (see [Quality Control](#quality-control)) and the problematic registrations were listed in [exclude.yml](https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/blob/main/exclude.yml). The processing was run again to generate the soft segmentations. 

Specify the path of preprocessed dataset with the flag `-path-data`. 

### Launch processing

```
cd processing_spine_generic
sct_run_batch -jobs -1 -path-data <PATH_DATA> -path-output <PATH-OUTPUT> -script process_data.sh -script-args exclude.yml
```

or use a config file:

```
config_process_data.json: 
{
  "path_data"   : "~/data_nvme_sebeda/datasets/data-multi-subject/",
  "path_output" : "~/data_nvme_sebeda/data_processed_sg_2023-08-04_NO_CROP",
  "script"      : "process_data.sh",
  "jobs"        : 50,
  "exclude_list":  ["sub-brnoUhb02", "sub-brnoUhb03", "sub-brnoUhb07", "sub-brnoUhb08", "sub-brnoUhb08", "sub-brnoUhb08", "sub-ucdavis01", "sub-ucdavis02", "sub-ucdavis03", "sub-ucdavis04", "sub-ucdavis05", "sub-ucdavis06", "sub-ucdavis07", "sub-beijingVerio01", "sub-beijingVerio02", "sub-beijingVerio03", "sub-beijingVerio04", "sub-beijingGE01", "sub-beijingGE02", "sub-beijingGE03", "sub-beijingGE04", "sub-ubc01", "sub-oxfordOhba02"]
}
```
```
sct_run_batch -config config_process_data.json
```

A `process_data_clean` folder is created in <PATH-OUTPUT> where the cropped data and derivatives are included. Here, only the images that have a manual segmentation and soft segmentation are transfered.

### Quality control

After running the analysis, check your Quality Control (qc) report by opening the file <PATH-OUTPUT>/qc/index.html. Use the "search" feature of the QC report to quickly jump to segmentations or labeling issues.

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

## Training

### config_generator.py
The script helps create joblibs that are going to represent splits of our dataset. It will create a <code>joblibs</code> folder containing the data split for each sub-experiment (i.e. hard_hard, soft_soft ...). The way we leverage the aforementioned python script is by running the bash script <code>utils/create_joblibs.sh</code> that will execute the following command for each sub-experiment:
```
python config_generator.py --config config_templates/hard_hard.json \
                           --datasets path/to/data
                           --ofolder path/to/joblib \
                           --contrasts T1w T2w T2star rec-average_dwi \
                           --seeds 15
```
in which one has to specify the config template for the sub-experiment, the dataset path, the joblibs output folder, the contrasts used for the experiment and the random generation seed(s) respectively.

### training_scripts
Once the joblibs describing how the data is split are generated, one can start training the different models within a sub-experiment. Notice that there are 3 folders in <code>training_scripts</code>, 2 of them are related to a specific MTS contrast and the last one is used to train models with the other contrasts. This flaw is due to the incompatibility of ivadomed's dataloader dealing with MTS contrasts properly, at the time of writing. We expect to address this problem in the next months so we can have a single bash script executing all the training experiments smoothly.
For clarity, we go over a few examples about how to use the current training scripts.
1. One wants to train MTS contrast-specific models. Then choose the right MTS contrast <code>acq-MTon_MTS</code> or <code>acq-T1w_MTS</code> and run the associated bash script. 
2. One wants to train contrast-specific (without MTS) models AND generalist models (including MTS) then run the bash script in <code>training_scripts/all/training_run.sh</code>.

All training runs are using the ivadomed's framework and logging training metrics in a <code>results</code> folder (optionally with wandb).

### inference.sh 
Once the models are trained, one can use the <code>evaluation/inference.sh</code> bash script to segment SC for tests participants and qualitatively analyze the results. Again like in all bash scripts mentioned in this project, one has to change a few parameters to adapt to one's environment (e.g. dataset path ...).

### Evaluation on spine-generic-multi-subject (MICCAI 2023)
Once the inference is done for all models and to reproduce the results presented in our paper, one would have to run the <code>compute_evaluation_metrics.py</code> after specifying the experiment folder paths inside that python script. A <code>spine-generic-test-results</code> folder will be created, in which a json file with the DICE and Relative Volume Difference (RVD) metrics for each experiments on the test set. To obtain the aggregated results **per_contrast** and **all_contrast**, run the <code>miccai_results_models.py</code> script. It generates aggregated results by the aforementioned category of models and the associated Latex table used in the paper. 

## Compute CSA on prediction masks

To compute CSA at C2-C3 vertebral levels on the prediction masks obtained from the trained models, the script `compute_csa.sh` is used. The input is the folder `data_processed_clean` (result from preprocessing) and the path of the prediction masks is added as an extra script argument `-script-args`.

For every trained model, you can run:

```
cd csa_evaluation
sct_run_batch -jobs -1 -path-data /data_processed_clean/ -path-output <PATH_OUTPUT> -script compute_csa_ivadomed.sh -script-args <PATH_PRED_MASKS>
```
The CSA results will be under `<PATH_OUTPUT>/results`.

To generate violin plots and analyse results, run the following command:

```
python gen_charts.py --contrasts T1w T2w T2star rec-average_dwi \
       --predictions_folder ../duke/projects/ivadomed/contrast-agnostic-seg/csa_measures_pred/group8-9_combined-2022-12-21/ \
       --baseline_folder ../duke/projects/ivadomed/contrast-agnostic-seg/archive_derivatives_softsegs-seg/contrast-agnostic-centerofmass-preprocess-clean-all-2022-10-22\results_MTS_renamed
```
## Run qc report on prediction masks

~~~
cd processing_other_datasets
sct_run_batch -path-data <PATH_DATA> -path-out <PATH-OUT> -script-args <PATH_PRED_MASK> -jobs 20 -script run_qc_prediction_ivadomed.sh
~~~
