## Procedure for training nnUNet on spine-generic data
This doc describes the procedure for training an nnUNet model on the contrast-agnostic dataset

### Converting the BIDS-formatted dataset to the nnUNet format

#### Step 1: Copying the BIDS dataset

The original BIDS dataset can be found at: `~/duke/projects/ivadomed/contrast-agnostic-seg/data_processed_sg_2023-03-10_NO_CROP/data_processed_clean`

Copy this dataset to your `home` directory. Note that this is the UNCROPPED version of the dataset. Hence, name the copied dataset accordingly. 


#### Step 2: Copying the joblib file

To compare the performance of ivadomed and nnUNet, the train/val/test splits of the dataset have to be the same. This information is contained in the `.joblib` file. This file can be found [here](https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/tree/ae/ohbm/config/miccai2023/data_split). Download the `split_datasets_all_seed=15.joblib` file to the same folder where the conversion script `convert_spine-generic_to_nnUNetv2.py` is located. 


#### Step 3: Running the dataset conversion script

Then, run the following command to convert the dataset to the appropriate format for nnUNet:

```python
python convert_spine-generic_to_nnUNetv2.py --path-data <path-to-the-copied-dataset>  --path-out ${nnUNet_raw} --path-joblib .  -dname spineGNoCropSoftAvgBin -dnum 713 --label-type soft
```

NOTES:
- `dname` - is the dataset name that nnUNet refers
- `dnum` or `dataset-number` - is the number associated with the dataset name. The name and number can be anything. 
- `label-type` - is the type of labels to be used for training. The original BIDS dataset has both binary (hard) and averaged (soft) labels. Setting this value to `soft` will use the averaged labels. Setting this value to `hard` will use the binary labels.


### Training and Testing the model

#### Step 1: Verifying dataset integrity

Before training the model, it is important to verify that the dataset was converted correctly and that nnUNet has no complaints about the conversion. To do so, run the following command:

```python
nnUNetv2_plan_and_preprocess -d <dnum> --verify_dataset_integrity
```
where `dnum` is the number associated with the dataset name.

This could take a while since nnUNet essentially saves the preprocessed data to disk so that training can be done faster.

#### Step 2: Training! 

To train the model, run the following command:

```python
CUDA_VISIBLE_DEVICES=X nnUNetv2_train 713 3d_fullres 0
```
This runs the full-resolution 3D model on fold 0 of the dataset. 

#### Step 3: Testing!

Once training is done, run the following command to test the model:

```python
CUDA_VISIBLE_DEVICES=X nnUNetv2_predict -i ${nnUNet_raw}/Dataset713_spineGNoCropSoftAvgBin/imagesTs -o <path-to-nnunet-folder>/nnUNet_results/Dataset713_spineGNoCropSoftAvgBin/test -d 713 -f 0 -c 3d_fullres
```

This commands runs the inference on the test set specified by the `imagesTs/` folder and saves the results in the `test/` folder in the results corresponding to the dataset. 


