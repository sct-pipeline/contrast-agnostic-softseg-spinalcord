## Instructions for running inference with the contrast-agnostic spinal cord segmentation model

The following steps are required for using the contrast-agnostic model. 

### Setting up the environment and Installing dependencies

The following commands show how to set up the environment. Note that the documentation assumes that the user has `conda` installed on their system. Instructions on installing `conda` can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. Create a conda environment with the following command:

```bash
conda create -n venv_monai python=3.9
```

2. Activate the environment with the following command:

```bash
conda activate venv_monai
```

3. The list of necessary packages can be found in `requirements_inference.txt`. Use the following command for installation:

```bash
pip install -r requirements_inference.txt
```

### Method 1: Running inference on a single image

The script for running inference is `run_inference_single_image.py`. Please run 
```
python run_inference_single_image.py -h
```
to get the list of arguments and their descriptions.


### Method 2: Running inference on a dataset (Advanced)

NOTE: This section is experimental and for advanced users only. Please use Method 1 for running inference.

#### Creating a datalist

The inference script assumes the dataset to be in Medical Segmentation Decathlon-style `json` file format containing image-label pairs. The `create_inference_msd_datalist.py` script allows to create one for your dataset. Use the following command to create the datalist:

```bash
python create_inference_msd_datalist.py --dataset-name spine-generic --path-data <path-to-dataset> --path-out <path-to-output-folder> --contrast-suffix T1w
```

`--dataset-name` - Corresponds to name of the dataset. The datalist will be saved as `<dname>_dataset.json`
`--path-data` - Path to the BIDS dataset
`--path-out` - Path to the output folder. The datalist will be saved under `<path-out>/<dname>_dataset.json`
`--contrast-suffix` - The suffix of the contrast to be used for pairing images/labels

> **Note**
> This script is not meant to run off-the-shelf. Placeholders are provided to update the script with the .... TODO


#### Running inference

Use the following command:

```bash
python run_inference.py --path-json <path/to/json> --chkp-path <path/to/checkpoint> --path-out <path/to/output> --model <unet/nnunet> --crop_size <48x160x320> --device <gpu/cpu>
```

`--path-json` - Path to the datalist created in Step 2
`--chkp-path` - Path to the model checkpoint. This folder should contain the `best_model_loss.ckpt`
`--path-out` - Path to the output folder where the predictions will be saved
`--model` - Model to be used for inference. Currently, only `unet` and `nnunet` are supported
`--crop_size` - Crop size used for center cropping the image before running inference. Recommended to be set to a multiple of 32
`--device` - Device to be used for inference. Currently, only `gpu` and `cpu` are supported



