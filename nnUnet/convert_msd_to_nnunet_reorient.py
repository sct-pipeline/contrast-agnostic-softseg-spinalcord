"""
This script takes a folder containing a list of datalist json files in MSD format dataset for each dataset used 
to train the contrast-agnostic model and converts it to the nnU-Net format (with reorientation to RPI).
Includes multiprocessing to spread the dataset conversion tasks across multiple workers

Example:
    python convert_msd_to_nnunet_reorient.py 
        -i /path/to/MSD/datalists/folder 
        -o /path/to/nnUNet_raw/folder 
        --taskname contrastAgnosticAllData 
        --tasknumber 716
        --workers 8

Author: Pierre-Louis Benveniste (adapted for multiprocessing by Naga Karthik)
"""

import os
import argparse
import json
from pathlib import Path
import tqdm
from collections import OrderedDict
from monai.data import load_decathlon_datalist
from multiprocessing import Pool, cpu_count


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MSD dataset to nnU-Net format')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the folder containing MSD json files')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory')
    parser.add_argument('--taskname', type=str, help='Name of the task', default='msLesionAgnostic')
    parser.add_argument('--tasknumber', type=int, required=True, help='Number of the task')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: number of CPU cores)')
    return parser.parse_args()


def process_single_image(args):
    """Process a single image and its corresponding label"""
    img_dict, counter, path_out_images, path_out_labels, taskname = args
    
    image_file_nnunet = os.path.join(path_out_images, f'{taskname}_{counter:03d}_0000.nii.gz')
    label_file_nnunet = os.path.join(path_out_labels, f'{taskname}_{counter:03d}.nii.gz')
    
    # Reorient image to RPI
    assert os.system(f"sct_image -i {img_dict['image']} -setorient RPI -o {image_file_nnunet}") == 0
    
    # Reorient label to RPI
    assert os.system(f"sct_image -i {img_dict['label']} -setorient RPI -o {label_file_nnunet}") == 0
    
    # Register label to image
    assert os.system(f"sct_register_multimodal -i {str(label_file_nnunet)} -d {str(image_file_nnunet)} "
                    f"-identity 1 -o {str(label_file_nnunet)} -owarp file_to_delete_{counter}.nii.gz "
                    f"-owarpinv file_to_delete_2_{counter}.nii.gz") == 0
    
    # Clean up temporary files
    os.system(f"rm file_to_delete_{counter}.nii.gz file_to_delete_2_{counter}.nii.gz")
    other_file_to_remove = str(label_file_nnunet).replace('.nii.gz', '_inv.nii.gz')
    os.system(f"rm {other_file_to_remove}")
    
    # Binarize label
    assert os.system(f"sct_maths -i {str(label_file_nnunet)} -bin 0.5 -o {str(label_file_nnunet)}") == 0
    
    return {
        'image': str(os.path.abspath(img_dict['image'])),
        'label': str(os.path.abspath(img_dict['label'])),
        'image_nnunet': image_file_nnunet,
        'label_nnunet': label_file_nnunet
    }


def process_dataset_parallel(data_list, path_out_images, path_out_labels, taskname, start_counter, num_workers):
    """Process a dataset in parallel using multiple workers"""
    with Pool(processes=num_workers) as pool:
        # Create work items list with all necessary arguments
        work_items = [
            (item, start_counter + i, path_out_images, path_out_labels, taskname)
            for i, item in enumerate(data_list)
        ]
        
        # Process items in parallel and show progress bar
        results = list(tqdm.tqdm(
            pool.imap(process_single_image, work_items),
            total=len(work_items),
            desc="Processing images"
        ))
    
    return results


def main():
    # Parse arguments
    args = parse_args()
    if args.workers is None:
        args.workers = cpu_count()
    
    # Define the output paths
    path_out = Path(os.path.join(args.output, f'Dataset{args.tasknumber}_{args.taskname}'))
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

    # Create directories
    for path in [path_out, path_out_imagesTr, path_out_imagesTs, path_out_labelsTr, path_out_labelsTs]:
        path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    datalists_list = [f for f in os.listdir(args.input) if f.endswith("_seed50.json")]
    train_data, val_data, test_data = [], [], []
    for datalist in sorted(datalists_list):
        print(f"Loading dataset: {datalist}")
        train_data += load_decathlon_datalist(os.path.join(args.input, datalist), True, "train")
        val_data += load_decathlon_datalist(os.path.join(args.input, datalist), True, "validation")
        test_data += load_decathlon_datalist(os.path.join(args.input, datalist), True, "test")

    print(f"Processing {len(datalists_list)} datasets with {args.workers} workers...")
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(val_data)}")
    print(f"Number of testing samples: {len(test_data)}")

    # Process training data (including validation)
    print("Processing training data...")
    train_results = process_dataset_parallel(
        train_data + val_data,
        path_out_imagesTr,
        path_out_labelsTr,
        args.taskname,
        1,
        args.workers
    )

    # Process test data
    print("Processing test data...")
    test_results = process_dataset_parallel(
        test_data,
        path_out_imagesTs,
        path_out_labelsTs,
        args.taskname,
        1,
        args.workers
    )

    # Create conversion dictionary
    conversion_dict = {}
    for result in train_results + test_results:
        conversion_dict[result['image']] = result['image_nnunet']
        conversion_dict[result['label']] = result['label_nnunet']

    # Save conversion dictionary
    with open(os.path.join(path_out, "conversion_dict.json"), "w") as f:
        json.dump(conversion_dict, f, indent=4)

    # Create dataset description
    json_dict = OrderedDict({
        'name': args.taskname,
        'description': args.taskname,
        'tensorImageSize': "3D",
        'reference': "TBD",
        'licence': "TBD",
        'release': "0.0",
        'channel_names': {
            "0": "MRI",
        },
        'labels': {
            "background": 0,
            "sc": 1,
        },
        'numTraining': len(train_results),
        'numTest': len(test_results),
        'file_ending': ".nii.gz",
        'image_orientation': "RPI",
        'training': [{'image': str(r['image_nnunet']), 'label': str(r['label_nnunet'])} for r in train_results],
        'test': [{'image': str(r['image_nnunet']), 'label': str(r['label_nnunet'])} for r in test_results]
    })

    # Save dataset description
    with open(os.path.join(path_out, "dataset.json"), "w") as f:
        json.dump(json_dict, f, indent=4)

    print("Conversion completed successfully!")


if __name__ == '__main__':
    main()