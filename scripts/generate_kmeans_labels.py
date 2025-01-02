"""
Script to generate labels using K-Means clustering of the T1w MRI scans of the spinal cord to be used for training SynthSeg.
"""

import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from loguru import logger
import os
import yaml
from datetime import datetime
from tqdm import tqdm


def normalize_intensity(data, min_in, max_in, min_out, max_out, mode="minmax", clip=True):
    """
    Normalize intensity of data in `data` with `keys`
    """
    if mode == "minmax":
        data = (data - min_in) / (max_in - min_in) * (max_out - min_out) + min_out
    elif mode == "meanstd":
        data = (data - min_in) / max_in
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if clip:
        data = np.clip(data, min_out, max_out)

    return data


def cluster_mri_with_mask(path_data, path_sc_seg, n_clusters=10, output_path=None):
    # Load the NIfTI image and mask
    img = nib.load(path_data)
    seg_sc = nib.load(path_sc_seg)
    
    img_data = img.get_fdata()
    seg_sc_data = seg_sc.get_fdata()
    
    # Reshape to 2D
    original_shape = img_data.shape
    img_data_2d = img_data.reshape(-1, 1)
    seg_sc_data_2d = seg_sc_data.reshape(-1)

    # Create a mask for non-zero intensity voxels
    non_zero_mask = (img_data_2d > 0).reshape(-1)    
    
    # Get indices for voxels to cluster (non-zero intensity AND not in spinal cord mask)
    cluster_indices = np.where((non_zero_mask) & (seg_sc_data_2d == 0))[0]

    # get non-mask voxels
    img_data_2d = img_data_2d[cluster_indices]

    # Standardize and cluster only non-mask voxels
    min_in, max_in = np.min(img_data_2d), np.max(img_data_2d)
    img_data_2d_norm = normalize_intensity(img_data_2d, min_in, max_in, min_out=0.0, max_out=1.0, mode="minmax", clip=True)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(img_data_2d_norm)
    
    # Create final label array
    # NOTE: if you set n_clusters=10, your final label values will be: 
    # 0: Background; 
    # 1: Spinal cord (from mask) 
    # 2-11: Other tissue clusters
    final_labels = np.zeros_like(seg_sc_data_2d)  # Background = 0
    final_labels[seg_sc_data_2d == 1] = 1  # Spinal cord mask = 1
    final_labels[cluster_indices] = clusters + 2  # Cluster labels start from 2

    # # Create final label array
    # final_labels = np.zeros_like(seg_sc_data_2d)
    # final_labels[seg_sc_data_2d == 1] = n_clusters  # Assign mask voxels highest label
    # final_labels[non_mask_indices] = clusters  # Assign cluster labels to non-mask voxels
    
    # Reshape back to 3D
    label_mask = final_labels.reshape(original_shape)
        
    return label_mask, kmeans.cluster_centers_


def main():

    path_root = "/home/GRAMES.POLYMTL.CA/u114716/duke/temp/sebeda/data_processed_sg_2023-08-24_NO_CROP/data_processed_clean"
    
    path_logs = "/home/GRAMES.POLYMTL.CA/u114716/synthseg-experiments/logs"
    if not os.path.exists(path_logs):
        os.makedirs(path_logs, exist_ok=True)
    timestamp = datetime.now().strftime(f"%Y%m%d-%H%M")
    logger.add(f"{path_logs}/generate_kmeans_labels_{timestamp}.log")
    
    path_datasplit_yaml = "/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/unused_but_useful_code/datasplit_all_soft_bin_seed15_ivado_joblib.yaml"
    # load the yaml file
    with open(path_datasplit_yaml, 'r') as file:
        datasplit = yaml.load(file, Loader=yaml.FullLoader)
    
    # get train and val keys
    train_val_subjects = datasplit["train"] + datasplit["val"]
    # train_val_subjects = train_val_subjects[5:7]  # for testing
    available_subjects = os.listdir(path_root)
    available_subjects = [subject for subject in available_subjects if subject.startswith("sub-")]
    
    train_val_subjects = sorted(list(set(train_val_subjects).intersection(available_subjects)))

    # contrasts = ["T1w", "T2w", "T2star", "flip-1_mt-on_MTS", "flip-2_mt-off_MTS", "rec-average_dwi"]
    contrasts = ["T1w"]  # synthseg paper mentions using only T1w for training
    clusters = np.arange(3, 11)

    path_out_root = "/home/GRAMES.POLYMTL.CA/u114716/synthseg-experiments/data_SynthSeg/training_label_maps_kmeans_t1w"
    if not os.path.exists(path_out_root):
        os.makedirs(path_out_root, exist_ok=True)

    for subject in tqdm(train_val_subjects, desc="Subjects"):

        for contrast in contrasts:

            if contrast == "rec-average_dwi":
                # define paths
                path_subject = f"{path_root}/{subject}/dwi"
                path_sc_seg_root = f"{path_root}/derivatives/labels/{subject}/dwi"
            else:
                # define paths
                path_subject = f"{path_root}/{subject}/anat"
                path_sc_seg_root = f"{path_root}/derivatives/labels/{subject}/anat"

            logger.info(f"Processing {subject} - {contrast}")

            path_data = os.path.join(path_subject, f"{subject}_{contrast}.nii.gz")
            path_sc_seg = os.path.join(path_sc_seg_root, f"{subject}_{contrast}_seg-manual.nii.gz")

            # load the image to get affine info later
            img = nib.load(path_data)

            for num_clusters in clusters:
                logger.info(f"\tclustering with K={num_clusters} ...")
                
                cluster_mask, cluster_centers = cluster_mri_with_mask(path_data, path_sc_seg, n_clusters=num_clusters)

                # Save output
                label_img = nib.Nifti1Image(cluster_mask, img.affine)
                output_path = f"{path_out_root}/{subject}_{contrast}_seg_K-{num_clusters}.nii.gz"
                nib.save(label_img, output_path)


if __name__ == "__main__":
    main()