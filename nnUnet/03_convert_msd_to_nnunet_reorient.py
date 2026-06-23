"""
Convert MSD datalists to nnU-Net format with RPI reorientation and sc_crop detection-based cropping.

Each image is reoriented to RPI, then the spinal cord is detected with sc_crop (YOLO-based, no GT
mask required) and both the image and label are cropped to the detected bounding box before saving
in nnU-Net format. The GT label is reduced to its largest connected component first, removing
isolated annotation-noise voxels. This ensures train/test consistency: inference also runs
sc_crop before nnUNet. A per-volume crop QC report (crop_qc_report.csv + crop_qc_summary.json)
is written alongside the dataset.

Example:
    python 03_convert_msd_to_nnunet_reorient.py \
        -i /path/to/MSD/datalists/folder \
        -o /path/to/nnUNet_raw/folder \
        --taskname ContrastAgnosticScCrop \
        --tasknumber 2000 \
        --workers 8

Author: Pierre-Louis Benveniste (adapted for multiprocessing by Naga Karthik; sc_crop by Quentin Revillon)
"""

import os
import argparse
import json
from pathlib import Path
import tqdm
from collections import OrderedDict
from multiprocessing import Pool, cpu_count

import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label
from sc_crop import detect, crop, check_label_crop, CropReport


def keep_largest_component(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return the label keeping only its largest 26-connected component.

    Isolated voxels far from the cord (annotation noise) artificially enlarge the
    detection-bbox QC check and add spurious labels to training. Removing them
    keeps a single clean spinal cord. Returns the input unchanged if it already
    has a single component.
    """
    data = np.asarray(nii.dataobj)
    lab, n = cc_label(data > 0, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return nii
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    cleaned = np.where(lab == sizes.argmax(), data, 0)
    return nib.Nifti1Image(cleaned, nii.affine, nii.header)


def force_orthonormal_affine(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Force direction cosines to be exactly orthonormal via SVD.

    Oblique acquisitions (e.g. sci-zurich sagittal T2w) have non-orthonormal
    direction cosines that survive sct_image -setorient RPI. SimpleITK/ITK
    (used by nnUNetv2_plan_and_preprocess) rejects these files.
    SVD gives the nearest orthonormal matrix while preserving spacing and origin.
    """
    affine = img.affine.copy()
    R = affine[:3, :3]
    spacing = np.linalg.norm(R, axis=0)
    U, _, Vt = np.linalg.svd(R)
    affine[:3, :3] = (U @ Vt) * spacing
    return nib.Nifti1Image(np.asarray(img.dataobj), affine, img.header)


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

    # Reorient image and label to RPI
    assert os.system(f"sct_image -i {img_dict['image']} -setorient RPI -o {image_file_nnunet}") == 0
    assert os.system(f"sct_image -i {img_dict['label']} -setorient RPI -o {label_file_nnunet}") == 0

    # sc_crop: detect SC bbox on image (no GT mask), crop image and label with the same bbox
    bbox = detect(nib.load(image_file_nnunet))
    nib.save(crop(nib.load(image_file_nnunet), bbox), image_file_nnunet)
    # Keep only the largest connected component of the GT: isolated parasite voxels
    # (annotation noise) otherwise stretch the bbox check and pollute training labels.
    label_nii = keep_largest_component(nib.load(label_file_nnunet))
    qc_result = check_label_crop(label_nii, bbox)
    nib.save(crop(label_nii, bbox), label_file_nnunet)

    # Put label to image to match dimension, resolution and orientation
    # '-identity 1': registration optimization (e.g. translations, rotations, deformations) is skipped
    assert os.system(f"sct_register_multimodal -i {str(label_file_nnunet)} -d {str(image_file_nnunet)} "
                    f"-identity 1 -o {str(label_file_nnunet)} -owarp file_to_delete_{counter}.nii.gz "
                    f"-owarpinv file_to_delete_2_{counter}.nii.gz") == 0

    # Clean up temporary files
    os.system(f"rm file_to_delete_{counter}.nii.gz file_to_delete_2_{counter}.nii.gz")
    other_file_to_remove = str(label_file_nnunet).replace('.nii.gz', '_inv.nii.gz')
    os.system(f"rm {other_file_to_remove}")

    # Binarize label
    assert os.system(f"sct_maths -i {str(label_file_nnunet)} -bin 0.5 -o {str(label_file_nnunet)}") == 0

    # Force orthonormal direction cosines — SimpleITK/ITK rejects oblique affines
    for path in [image_file_nnunet, label_file_nnunet]:
        nib.save(force_orthonormal_affine(nib.load(path)), path)

    return {
        'image': str(os.path.abspath(img_dict['image'])),
        'label': str(os.path.abspath(img_dict['label'])),
        'image_nnunet': image_file_nnunet,
        'label_nnunet': label_file_nnunet,
        'qc': qc_result,
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


def load_json_datalist(datalist_path, key_to_extract):
    """Load a json datalist file and extract the specified key"""
    with open(datalist_path, 'r') as f:
        datalist = json.load(f)
    return datalist[key_to_extract]


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
        train_data += load_json_datalist(os.path.join(args.input, datalist), key_to_extract="train")
        val_data += load_json_datalist(os.path.join(args.input, datalist), key_to_extract="validation")
        test_data += load_json_datalist(os.path.join(args.input, datalist), key_to_extract="test")

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

    # Save crop QC report
    report = CropReport()
    for result in train_results + test_results:
        report.add(result['label'], result['qc'])
    report.save(os.path.join(path_out, "crop_qc_report.csv"))
    report.save_summary(os.path.join(path_out, "crop_qc_summary.json"))

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