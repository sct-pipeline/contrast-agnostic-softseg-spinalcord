# contrast-agnostic-softseg-spinalcord
Contrast-agnostic spinal cord segmentation project with softseg

This repo creates a series of preparations for comparing the newly trained ivadomed models (Pytorch based), with the old models that are currently implemented in spinal cord toolbox [SCT] (tensorflow based).

## Dependencies

- [SCT 5.3.0](https://github.com/neuropoly/spinalcordtoolbox/releases/tag/5.3.0)
- Python 3.7.

## Dataset and preprocessing

The original data are from the [spine-generic multi-subject](https://github.com/spine-generic/data-multi-subject/releases/tag/r20220125)

These data were preprocessed to segment the spinal cord, as described in [this PR](https://github.com/spine-generic/spine-generic/pull/249).
The output of these data are stored internally at NeuroPoly's git-annex `data` server under `spine-generic-processed`.

These processed dataset then need to be input into [another processing scripts](https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/blob/main/process_data.sh) that:
- creates a mask around the T2 spinal cord
- co-register all contrasts to the T2 spinal cord 
- average all segmentations from each contrast within the same space (the T2)
- bring back the segmentations to the original image space of each contrast (except for the T2)

The output of this script is a new 'derivatives/labels_softseg/' folder that contains the soft labels to be used in this contrast-agnostic segmentation project. 

Specify the path of preprocessed dataset with the flag `-path-data`.

Launch processing:

```
sct_run_batch -jobs -1 -path-data <PATH_DATA> -path-output <PATH-OUTPUT> -script process_data.sh
```


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
