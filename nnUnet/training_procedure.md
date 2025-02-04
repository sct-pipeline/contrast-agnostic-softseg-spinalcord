## Procedure for training nnUNet on spine-generic data
This document describes the procedure for training the contrast-agnostic spinal cord segmentation model. The model is trained using the nnUNet framework.

### Creating the dataset

#### Step 1: Creating MSD-style json files from BIDS datasets

* Before running the script to create json files in MSD format (Medical Segmentation Decathlon), ensure that the datasets are downloaded using `git-annex` and located in your `home` directory. 
* To reproduce results using the same dataset as the contrast-agnostic model, run `bash scripts/batch_create_datalists.sh` script to create dataset json files for each dataset. The list of datasets is specified as a variable inside the script. 

TODO: update the dataset creation script to directly download the data from `git-annex` instead of user having to download the data manually.


#### Step 2: Converting the dataset to nnUNet format

Once the dataset json files are created, run the following command to convert the dataset to nnUNet format:

```python

python dataset_conversion/convert_msd_to_nnunet_reorient.py -i ~/path/to/folder/containing/datalist/jsons -o ~/path/to/nnUNet_raw --taskname ContrastAgnosticAll --tasknumber 716 --workers 16

```

This creates a new folder in the `nnUNet_raw` directory with the name `Dataset716_ContrastAgnosticAll`. The `workers` argument specifies the number of workers used for parallel processing.


### Training and Testing the model

#### Step 1: Verifying dataset integrity and Preprocessing

Before training the model, it is important to verify that the dataset was converted correctly and that nnUNet has no complaints about the conversion. To do so, run the following command:

```python
nnUNetv2_plan_and_preprocess -d -c 2d 3d_fullres --verify_dataset_integrity
```

Note that this command checks whether nnunet likes the dataset format and also runs preprocessing for both 2D and 3D variants of the model (which might take a while).

#### Step 2: Training! 

To train the model, run the following command:

```python
CUDA_VISIBLE_DEVICES=X nnUNetv2_train 716 <2d/3d_fullres> 0
```
This runs the 2D or 3D model on fold 0 of the dataset. 

#### Step 3: Testing!

Once training is done, run the following command to test the model:

```python
CUDA_VISIBLE_DEVICES=X nnUNetv2_predict -i ${nnUNet_raw}/Dataset716_ContrastAgnosticAll/imagesTs -o <path-to-nnunet-folder>/nnUNet_results/Dataset716_ContrastAgnosticAll/test -d 716 -f 0 -c <2d/3d_fullres>
```

This commands runs the inference on the test set specified by the `imagesTs/` folder and saves the results in the `test/` folder in the results corresponding to the dataset. 

TODO: add script used to train the model on compute canada cluster
