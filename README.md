# Contrast-agnostic soft segmentation of the spinal cord
Contrast-agnostic spinal cord segmentation project with softseg

This repo creates a series of preparations for comparing the newly trained ivadomed models (Pytorch based), with the old models that are currently implemented in spinal cord toolbox [SCT] (tensorflow based).

## Dependencies

- [SCT 5.3.0](https://github.com/neuropoly/spinalcordtoolbox/releases/tag/5.3.0)
- Python 3.7.

## Dataset
The data are from the [spine-generic multi-subject](https://github.com/spine-generic/data-multi-subject/)

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

Next steps are to generate a constrast-agnostic soft segmentation: 
- creates spinal cord segmentation (if it doesn't exist yet)
- creates a mask around the T2 spinal cord
- co-register all contrasts to the T2 spinal cord 
- average all segmentations from each contrast within the same space (the T2)
- bring back the segmentations to the original image space of each contrast (except for the T2)
- Crop images, segmentations and soft segmentation arround the spinal cord

The output of this script is a new 'derivatives/labels_softseg/' folder that contains the soft labels to be used in this contrast-agnostic segmentation project. All the registration were manually QC-ed and the problematic registrations were listed in [exclude_reg.yml](https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/blob/main/exclude_reg.yml). The processing was run again to generate the soft segmentations. 

Specify the path of preprocessed dataset with the flag `-path-data`. 

### Launch processing

```
sct_run_batch -jobs -1 -path-data <PATH_DATA> -path-output <PATH-OUTPUT> -script process_data.sh -script-args exclude_reg.yml
```

A `process_data_clean` folder is created in <PATH-OUTPUT> where the cropped data and derivatives are included. Here, only the images that have a manual segmentation and soft segmentation are transfered.

### Quality control

After running the analysis, check your Quality Control (qc) report by opening the file <PATH-OUTPUT>/qc/index.html. Use the "search" feature of the QC report to quickly jump to segmentations or labeling issues.

**1. Segmentations**

If segmentation issues are noticed while checking the quality report, proceed to manual correction using the procedure below:

* In QC report, search for "deepseg" to only display results of spinal cord segmentation
* Review segmentation and spinal cord.
* Click on the F key to indicate if the segmentation is OK ✅, needs manual correction ❌ or if the data is not usable ⚠️ (artifact). Two .yml lists, one for manual corrections and one for the unusable data, will automatically be generated.
* Download the lists by clicking on **Download QC Fails*** and on **Download Qc Artifacts**.

Proceed to manual correction using FSLeyes or ITK snap. Uplaod the manual segmentations (_seg-manual.nii.gz) with json sidecar in the derivatives.
Re-run the analysis: [Launch processing](#launch-processing)

**2. Registrations**

* In QC report, search for "sct_register_multimodal" to only display results of registration.
* Click on the F key to indicate if the registration is OK ✅, needs to be excluded ❌ or if the data is not usable ⚠️ (artifact). Two .yml lists, will automatically be generated.
* Downlaod the list by clicking on **Download QC Fails** and add the file names under `FILES_REG` to https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/blob/main/exclude_reg.yml
* Re-run the analysis: [Launch processing](#launch-processing)

**3. Soft segmentations**

* In QC report, search for "softseg" to only display results of spinal cord soft segmentation
* Click on the F key to indicate if the soft segmentation is OK ✅, needs manual correction ❌ or if the data is not usable ⚠️ (artifact). Two .yml lists, one for manual corrections and one for the unusable data, will automatically be generated.
* Downlaod the list by clicking on **Download QC Fails**.
* Uplaod the soft segmentations under `data-multi-subject/darivatives/labels_softseg`.
* Re-run the analysis: [Launch processing](#launch-processing)

## Training

### create_training_joblib.py
The function creates a joblib that allocates data from the testing set of the SCT model to the testing set of the ivadomed model. The output (new_splits.joblib) needs to be assigned on the config.json in the field "split_dataset": {"fname_split": new_splits.joblib"}. 
Multiple datasets (BIDS folders) can be used as input for the creation of the joblib. The same list should be assigned on the config.json file in the path_data field.


### compare_with_sct_model.py
The comparison is being done by running `sct_deepseg_sc` on every subject/contrast that was used in the testing set on ivadomed.

One thing to note, is that the SCT scores have been marked after the usage of the function `sct_get_centerline` and cropping around this prior.
In order to make a fair comparison, the ivadomed model needs to be tested on a testing set that has the centerline precomputed.

The function `compare_with_sct_model.py` prepares the dataset for this comparison by using `sct_get_centerline` on the images and using this prior on the TESTING set.

The output folder will contain as many folders as inputs are given to `compare_with_sct_model.py`, with the suffix SCT. These folders "siumulate" output folders from ivadomed (they contain  evaluation3dmetrics.csv files) in order to use violinpolots visualizations from the script `visualize_and_compare_testing_models.py`



Problems with this approach: 
1. _centerline.nii.gz derivatives for the testing set files are created in the database
2. The order that processes need to be done might confuse people a bit:
    i. Joblib needs to be created
    ii. The ivadomed model needs to be trained
    iii. compare_with_sct_model script needs to run
    iv. The ivadomed model needs to be tested 
