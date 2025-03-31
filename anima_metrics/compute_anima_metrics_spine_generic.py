"""
This script evaluates the reference segmentations and model predictions 
using the "animaSegPerfAnalyzer" command

****************************************************************************************
SegPerfAnalyser (Segmentation Performance Analyzer) provides different marks, metrics 
and scores for segmentation evaluation.
3 categories are available:
    - SEGMENTATION EVALUATION:
        Dice, the mean overlap
        Jaccard, the union overlap
        Sensitivity
        Specificity
        NPV (Negative Predictive Value)
        PPV (Positive Predictive Value)
        RVE (Relative Volume Error) in percentage
    - SURFACE DISTANCE EVALUATION:
        Hausdorff distance
        Contour mean distance
        Average surface distance
    - DETECTION LESIONS EVALUATION:
        PPVL (Positive Predictive Value for Lesions)
        SensL, Lesion detection sensitivity
        F1 Score, a F1 Score between PPVL and SensL

Results are provided as follows: 
Jaccard;    Dice;   Sensitivity;    Specificity;    PPV;    NPV;    RelativeVolumeError;    
HausdorffDistance;  ContourMeanDistance;    SurfaceDistance;  PPVL;   SensL;  F1_score;       

NbTestedLesions;    VolTestedLesions;  --> These metrics are computed for images that 
                                            have no lesions in the GT
****************************************************************************************

Mathematical details on how these metrics are computed can be found here:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6135867/pdf/41598_2018_Article_31911.pdf

and in Section 4 of this paper (for how the subjects with no lesions are handled):
https://portal.fli-iam.irisa.fr/files/2021/06/MS_Challenge_Evaluation_Challengers.pdf

INSTALLATION:
##### STEP 0: Install git lfs via apt if you don't already have it.
##### STEP 1: Install ANIMA #####
cd ~
mkdir anima/
cd anima/
wget -q https://github.com/Inria-Visages/Anima-Public/releases/download/v4.2/Anima-Ubuntu-4.2.zip   (change version to latest)
unzip Anima-Ubuntu-4.2.zip
git lfs install
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Public.git
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Data-Public.git
cd ~
mkdir .anima/
touch .anima/config.txt
nano .anima/config.txt

##### STEP 2: Configure directories #####
# Variable names and section titles should stay the same
# Put this file in ${HOME}/.anima/config.txt
# Make the anima variable point to your Anima public build
# Make the extra-data-root point to the data folder of Anima-Scripts
# The last folder separator for each path is crucial, do not forget them
# Use full paths, nothing relative or using tildes 

[anima-scripts]
anima = ${HOME}/anima/Anima-Binaries-4.2/
anima-scripts-public-root = ${HOME}/anima/Anima-Scripts-Public/
extra-data-root = ${HOME}/anima/Anima-Scripts-Data-Public/

USAGE:
python compute_anima_metrics.py --pred_folder <path_to_predictions_folder> 
--gt_folder <path_to_gt_folder> -dname <dataset_name> 


NOTE 1: For checking all the available options run the following command from your terminal: 
      <anima_binaries_path>/animaSegPerfAnalyzer -h
NOTE 2: We use certain additional arguments below with the following purposes:
      -i -> input image, -r -> reference image, -o -> output folder
      -d -> evaluates surface distance, -l -> evaluates the detection of lesions
      -a -> intra-lesion evalulation (advanced), -s -> segmentation evaluation, 
      -X -> save as XML file  -A -> prints details on output metrics and exits

Authors: Naga Karthik, Jan Valosek
"""

import os
import glob
import subprocess
import argparse
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import nibabel as nib

STANDARD_DATASETS = ["spine-generic"]

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Compute ANIMA metrics using animaSegPerfAnalyzer')

    # Arguments for model, data, and training
    parser.add_argument('--pred-folder', required=True, type=str,
                        help='Path to the folder containing nifti images of test predictions AND GTs'
                        ' when method == monai, else path to the xml files containing the pre-computed'
                        ' ANIMA metrics when method == [deepseg*, propseg]')
    parser.add_argument('-dname', '--dataset-name', required=True, type=str, choices=STANDARD_DATASETS,
                        help='Dataset name used for storing on git-annex. For region-based metrics, '
                             'append "-region" to the dataset name. Default: spine-generic')
    parser.add_argument('--method', required=True, type=str, default='monai', 
                        choices=['monai', 'synthseg', 'deepseg2d', 'deepseg3d', 'propseg', 'v20', 'v30'],
                        help='Segmentation method to compute metrics for. Default: monai')

    return parser


def get_test_metrics_by_dataset(pred_folder, output_folder, anima_binaries_path, data_set, method):
    """
    Computes the test metrics given folders containing nifti images of test predictions 
    and GT images by running the "animaSegPerfAnalyzer" command
    """
    if method == 'v20':
        pred_suffix = 'seg_v20' # '_pred.nii.gz'
    elif method == 'v30':
        pred_suffix = 'seg_nnunet-AllRandInit3D_bin'
    elif method == 'deepseg2d':
        pred_suffix = 'seg_deepseg_2d'
    gt_suffix = 'softseg_bin' # '_softseg_gt.nii.gz'
    if data_set in STANDARD_DATASETS:
        # glob all the predictions and GTs and get the last three digits of the filename
        pred_files = sorted(glob.glob(f"{pred_folder}/**/**/*_{pred_suffix}.nii.gz"))
        gt_files = sorted(glob.glob(f"{pred_folder}/**/**/*_{gt_suffix}.nii.gz"))

        # dataset_name_nnunet = fetch_filename_details(pred_files[0])[0]

        # loop over the predictions and compute the metrics
        for pred_file, gt_file in zip(pred_files, gt_files):
            
            subject_contrast_pred = pred_file.split('/')[-1].replace(f'_{pred_suffix}.nii.gz', '')
            subject_contrast_gt = gt_file.split('/')[-1].replace(f'_{gt_suffix}.nii.gz', '')

            # make sure the subject and session IDs match
            print(f"Subject_Contrast for Preds and GTs: {subject_contrast_pred}, {subject_contrast_gt}")
            assert subject_contrast_pred == subject_contrast_gt, 'Subject_Contrast for Preds and GTs do not match. Please check the filenames.'
            
            # load the predictions and GTs
            pred_npy = nib.load(pred_file).get_fdata()
            gt_npy = nib.load(gt_file).get_fdata()
            
            # make sure the predictions are binary because ANIMA accepts binarized inputs only
            pred_npy = np.array(pred_npy > 0.5, dtype=float)
            gt_npy = np.array(gt_npy > 0.5, dtype=float)

            # Save the binarized predictions and GTs
            pred_nib = nib.Nifti1Image(pred_npy, affine=np.eye(4))
            gtc_nib = nib.Nifti1Image(gt_npy, affine=np.eye(4))
            nib.save(img=pred_nib, filename=os.path.join(pred_folder, f"{subject_contrast_pred}_pred_bin.nii.gz"))
            nib.save(img=gtc_nib, filename=os.path.join(pred_folder, f"{subject_contrast_gt}_gt_bin.nii.gz"))
            # exit()

            # Run ANIMA segmentation performance metrics on the predictions            
            # skip lesion evaluation metrics, just do segmentation evaluation
            seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -s -X'

            os.system(seg_perf_analyzer_cmd %
                        (os.path.join(anima_binaries_path, 'animaSegPerfAnalyzer'),
                        os.path.join(pred_folder, f"{subject_contrast_pred}_pred_bin.nii.gz"),
                        os.path.join(pred_folder, f"{subject_contrast_gt}_gt_bin.nii.gz"),
                        os.path.join(output_folder, f"{subject_contrast_pred}")))

            # Delete temporary binarized NIfTI files
            os.remove(os.path.join(pred_folder, f"{subject_contrast_pred}_pred_bin.nii.gz"))
            os.remove(os.path.join(pred_folder, f"{subject_contrast_gt}_gt_bin.nii.gz"))

        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_filepaths = [os.path.join(output_folder, f) for f in
                                os.listdir(output_folder) if f.endswith('.xml')]
        
        return subject_filepaths


def main():

    # get the ANIMA binaries path
    cmd = r'''grep "^anima = " ~/.anima/config.txt | sed "s/.* = //"'''
    anima_binaries_path = subprocess.check_output(cmd, shell=True).decode('utf-8').strip('\n')
    print('ANIMA Binaries Path:', anima_binaries_path)
    # version = subprocess.check_output(anima_binaries_path + 'animaSegPerfAnalyzer --version', shell=True).decode('utf-8').strip('\n')
    print('Running ANIMA version:',
          subprocess.check_output(anima_binaries_path + 'animaSegPerfAnalyzer --version', shell=True).decode(
              'utf-8').strip('\n'))

    parser = get_parser()
    args = parser.parse_args()

    # define variables
    pred_folder = args.pred_folder
    dataset_name = args.dataset_name
    method = args.method

    output_folder = os.path.join(pred_folder, f"anima_stats_{method}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    print(f"Saving ANIMA performance metrics to {output_folder}")

    # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
    if method in ['monai', 'synthseg', 'v20', 'deepseg2d', 'v30']:
        subject_filepaths = get_test_metrics_by_dataset(pred_folder, output_folder, anima_binaries_path, 
                                                        data_set=dataset_name, method=method)
    elif method in ['deepseg3d', 'propseg']:
        subject_filepaths = sorted(glob.glob(f"{pred_folder}/*.xml"))
    else:
        raise ValueError(f"Unknown method {method}!")

    # define a dictionary to store the metrics for each subject
    test_metrics = {}

    # Update the test metrics dictionary by iterating over all subjects
    for subject_filepath in subject_filepaths:
        subject = os.path.split(subject_filepath)[-1].split('_')[0]
        contrast = os.path.split(subject_filepath)[-1].split('_')[1]
        root_node = ET.parse(source=subject_filepath).getroot()

        # create a dictionary to store the metrics for each subject
        if subject not in test_metrics.keys():
            test_metrics[subject] = {}
        
        # create a dictionary to store the metrics for each contrast
        test_metrics[subject].update({contrast: defaultdict(list)})

        # if GT is empty then metrics aren't calculated, hence the only entries in the XML file 
        # NbTestedLesions and VolTestedLesions, both of which are zero. Hence, we can skip subjects
        # with empty GTs by checked if the length of the .xml file is 2
        if len(root_node) == 2:
            print(f"Skipping Subject={int(subject):03d} ENTIRELY Due to Empty GT!")
            continue
        
        for metric in list(root_node):
            name, value = metric.get('name'), float(metric.text)

            if np.isinf(value) or np.isnan(value):
                # print(f'Skipping Metric={name} for Subject={int(subject):03d} Due to INF or NaNs!')
                print(f'Skipping Metric={name} for Subject={subject} Due to INF or NaNs!')
                continue
            
            # update the test metrics dictionary
            test_metrics[subject][contrast][name].append(value)

    # print(test_metrics)
    # Print aggregation of each metric via mean and standard dev.
    with open(os.path.join(output_folder, f'log_{dataset_name}.txt'), 'a') as f:
        print('Test Phase Metrics [ANIMA]: ', file=f)

    # get the list of contrasts
    contrasts = sorted(list(test_metrics[list(test_metrics.keys())[0]].keys()))
    print(f"Contrasts: {contrasts}")
    metrics = ['Dice', 'RelativeVolumeError', 'SurfaceDistance', 'HausdorffDistance']

    metrics_per_contrast = {}
    metrics_avg_all = {}

    for metric in metrics:
        metrics_per_contrast[metric] = {}
        metrics_avg_all[metric] = []
        with open(os.path.join(output_folder, f'log_{dataset_name}.txt'), 'a') as f:
            print(f"\t{metric}:", file=f)

        for contrast in contrasts:
            metrics_per_contrast[metric][contrast] = []
            for subject in test_metrics.keys():
                # if subject does not have contrast, skip
                if contrast not in test_metrics[subject].keys():
                    print(f"Skipping Subject={subject}'s Contrast={contrast} as it is missing!")
                else:
                    metrics_per_contrast[metric][contrast].extend(test_metrics[subject][contrast][metric])
                    metrics_avg_all[metric].extend(test_metrics[subject][contrast][metric])

            with open(os.path.join(output_folder, f'log_{dataset_name}.txt'), 'a') as f:
                print(f"\t\t{contrast} --> Mean ± Std: {np.mean(metrics_per_contrast[metric][contrast]):0.3f} ± {np.std(metrics_per_contrast[metric][contrast]):0.3f}", file=f)

        with open(os.path.join(output_folder, f'log_{dataset_name}.txt'), 'a') as f:
            print(f"\t---------------------------------------------", file=f)
            print(f"\tOverall (N={len(metrics_avg_all[metric])}) --> Mean ± Std: {np.mean(metrics_avg_all[metric]):0.3f} ± {np.std(metrics_avg_all[metric]):0.3f}", file=f)
            print(f"\t---------------------------------------------", file=f)


if __name__ == '__main__':
    main()
