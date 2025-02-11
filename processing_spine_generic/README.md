
This document describes the procedure for preprocessing the spine-generic dataset which was used for training the [SoftSeg-based contrast-agnostic model](https://www.sciencedirect.com/science/article/pii/S1361841525000210). Future versions of the model have also been trained on the preprocessed dataset.


## Dataset
The source data can be found at [spine-generic multi-subject](https://github.com/spine-generic/data-multi-subject/).

The preprocessed data are located at [spine-generic multi-subject/derivatives/data_preprocessed](https://github.com/spine-generic/data-multi-subject/tree/master/derivatives/data_preprocessed)

The GT used for training are located at: [spine-generic multi-subject/derivatives/labels_softseg_bin](https://github.com/spine-generic/data-multi-subject/tree/master/derivatives/labels_softseg_bin)


## Preprocessing
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
