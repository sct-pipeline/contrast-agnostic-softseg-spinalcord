import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import torch
import subprocess
import os
import pandas as pd
import re
import importlib
import pkgutil
from batchgenerators.utilities.file_and_folder_operations import *


CONTRASTS = {
    "t1map": ["T1map"],
    "mp2rage": ["inv-1_part-mag_MP2RAGE", "inv-2_part-mag_MP2RAGE"],
    "t1w": ["T1w", "space-other_T1w", "acq-lowresSag_T1w"],
    "t2w": ["T2w", "space-other_T2w", "acq-lowresSag_T2w", "acq-highresSag_T2w"],
    "t2star": ["T2star", "space-other_T2star"],
    "dwi": ["rec-average_dwi", "acq-dwiMean_dwi"],
    "mt-on": ["flip-1_mt-on_space-other_MTS", "acq-MTon_MTR"],
    "mt-off": ["flip-2_mt-off_space-other_MTS"],
    "unit1": ["UNIT1"],
    "psir": ["PSIR"],
    "stir": ["STIR"]
}

def get_pathology_wise_split(unified_df):
    
    # ===========================================================================
    #                Subject-wise Pathology split
    # ===========================================================================
    pathologies = unified_df['pathologyID'].unique()

    # count the number of subjects for each pathology
    pathology_subjects = {}
    for pathology in pathologies:
        pathology_subjects[pathology] = len(unified_df[unified_df['pathologyID'] == pathology]['subjectID'].unique())

    # merge MildCompression, DCM, MildCompression/DCM into DCM
    pathology_subjects['DCM'] = pathology_subjects['MildCompression'] + pathology_subjects['MildCompression/DCM'] + pathology_subjects['DCM']
    pathology_subjects.pop('MildCompression', None)
    pathology_subjects.pop('MildCompression/DCM', None)

    # ===========================================================================
    #                Contrast-wise Pathology split
    # ===========================================================================
    # for a given contrast, count the number of images for each pathology
    pathology_contrasts = {}
    for contrast in CONTRASTS.keys():
        pathology_contrasts[contrast] = {}
        # initialize the count for each pathology
        pathology_contrasts[contrast] = {pathology: 0 for pathology in pathologies}
        for pathology in pathologies:
                pathology_contrasts[contrast][pathology] += len(unified_df[(unified_df['pathologyID'] == pathology) & (unified_df['contrastID'] == contrast)]['filename'])

    # merge MildCompression, DCM, MildCompression/DCM into DCM
    for contrast in pathology_contrasts.keys():
        pathology_contrasts[contrast]['DCM'] = pathology_contrasts[contrast]['MildCompression'] + pathology_contrasts[contrast]['MildCompression/DCM'] + pathology_contrasts[contrast]['DCM']
        pathology_contrasts[contrast].pop('MildCompression', None)
        pathology_contrasts[contrast].pop('MildCompression/DCM', None)

    return pathology_subjects, pathology_contrasts
    

def get_datasets_stats(datalists_root, contrasts_dict, path_save):

    # create a unified dataframe combining all datasets
    csvs = [os.path.join(datalists_root, file) for file in os.listdir(datalists_root) if file.endswith('_seed50.csv')]
    unified_df = pd.concat([pd.read_csv(csv) for csv in csvs], ignore_index=True)
    
    # sort the dataframe by the dataset column
    unified_df = unified_df.sort_values(by='datasetName', ascending=True)

    # save the originals as the csv
    unified_df.to_csv(os.path.join(path_save, 'dataset_contrast_agnostic.csv'), index=False)

    # dropna
    unified_df = unified_df.dropna(subset=['pathologyID'])

    contrasts_final = list(contrasts_dict.keys())
    # rename the contrasts column as per contrasts_final
    for c in unified_df['contrastID'].unique():
        for cf in contrasts_final:
            if re.search(cf, c.lower()):
                unified_df.loc[unified_df['contrastID'] == c, 'contrastID'] = cf
                break

    # NOTE: MTon-MTR is same as flip-1_mt-on_space-other_MTS, but the naming is not mt-on
    # so doing the renaming manually
    unified_df.loc[unified_df['contrastID'] == 'acq-MTon_MTR', 'contrastID'] = 'mt-on'

    splits = ['train', 'validation', 'test']
    # count the number of images per contrast
    df = pd.DataFrame(columns=['contrast', 'train', 'validation', 'test'])
    for contrast in contrasts_final:
        df.loc[len(df)] = [contrast, 0, 0, 0]
        for split in splits:
            df.loc[df['contrast'] == contrast, split] = len(unified_df[(unified_df['contrastID'] == contrast) & (unified_df['split'] == split)])

    # sort the dataframe by the contrast column
    df = df.sort_values(by='contrast', ascending=True)
    # add a row for the total number of images
    df.loc[len(df)] = ['TOTAL', df['train'].sum(), df['validation'].sum(), df['test'].sum()]
    # add a column for total number of images per contrast
    df['#images_per_contrast'] = df['train'] + df['validation'] + df['test']
    

    # get the subject-wise pathology split
    pathology_subjects, pathology_contrasts = get_pathology_wise_split(unified_df)
    df_pathology = pd.DataFrame.from_dict(pathology_subjects, orient='index', columns=['Number of Subjects'])
    # rename index to Pathology
    df_pathology.index.name = 'Pathology'
    # sort the dataframe by the pathology column
    df_pathology = df_pathology.sort_index()
    # add a row for the total number of subjects
    df_pathology.loc['TOTAL'] = df_pathology['Number of Subjects'].sum()

    # create a txt file
    with open(os.path.join(path_save, 'dataset_stats_overall.txt'), 'w') as f:
        # 1. write the datalists used in a bullet list
        f.write(f"DATASETS USED FOR MODEL TRAINING (n={len(datalists)}):\n\n")
        for datalist in datalists:
            f.write(f"\t- {datalist}\n")
        f.write("\n\n")

        # 2. write the table with proper formatting
        f.write(f"SPLITS ACROSS DIFFERENT CONTRASTS (n={len(contrasts_final)}):\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        # 3. write the pathology-wise split
        f.write(f"\nPATHOLOGY-WISE SPLIT:\n\n")
        f.write(df_pathology.to_markdown())
        f.write("\n\n")

    # create a unified dataframe combining all datasets
    csvs = [os.path.join(datalists_root, file) for file in os.listdir(datalists_root) if file.endswith('.csv')]
    unified_df = pd.concat([pd.read_csv(csv) for csv in csvs], ignore_index=True)
    
    # sort the dataframe by the dataset column
    unified_df = unified_df.sort_values(by='datasetName', ascending=True)

    # save as csv
    unified_df.to_csv(os.path.join(path_save, 'dataset_contrast_agnostic.csv'), index=False)


# Taken from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/find_class_by_name.py
def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Check if any label image patch is empty in the batch
def check_empty_patch(labels):
    for i, label in enumerate(labels):
        if torch.sum(label) == 0.0:
            # print(f"Empty label patch found at index {i}. Skipping training step ...")
            return None
    return labels  # If no empty patch is found, return the labels


def get_git_branch_and_commit(dataset_path=None):
    """
    :return: git branch and commit ID, with trailing '*' if modified
    Taken from: https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/utils/sys.py#L476 
    and https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/utils/sys.py#L461
    """

    # branch info
    b = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=dataset_path)
    b_output, _ = b.communicate()
    b_status = b.returncode

    if b_status == 0:
        branch = b_output.decode().strip()
    else:
        branch = "!?!"

    # commit info
    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=dataset_path)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=dataset_path)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        unclean = True
        for line in output.decode().strip().splitlines():
            line = line.rstrip()
            if line.startswith("??"):  # ignore ignored files, they can't hurt
                continue
            break
        else:
            unclean = False
        if unclean:
            commit += "*"

    return branch, commit


def dice_score(prediction, groundtruth):
    smooth = 1.
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()
    # loss = (2 * numer + self.smooth) / (denor + self.smooth)
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice


def plot_slices(image, gt, pred, debug=False):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    # bring everything to numpy
    image = image.numpy()
    gt = gt.numpy()
    pred = pred.numpy()

    if not debug:
        mid_sagittal = image.shape[2]//2
        # plot X slices before and after the mid-sagittal slice in a grid
        fig, axs = plt.subplots(3, 6, figsize=(10, 6))
        fig.suptitle('Original Image --> Ground Truth --> Prediction')
        for i in range(6):
            axs[0, i].imshow(image[:, :, mid_sagittal-3+i].T, cmap='gray'); axs[0, i].axis('off') 
            axs[1, i].imshow(gt[:, :, mid_sagittal-3+i].T); axs[1, i].axis('off')
            axs[2, i].imshow(pred[:, :, mid_sagittal-3+i].T); axs[2, i].axis('off')

        # fig, axs = plt.subplots(1, 3, figsize=(10, 8))
        # fig.suptitle('Original Image --> Ground Truth --> Prediction')
        # slice = image.shape[2]//2

        # axs[0].imshow(image[:, :, slice].T, cmap='gray'); axs[0].axis('off') 
        # axs[1].imshow(gt[:, :, slice].T); axs[1].axis('off')
        # axs[2].imshow(pred[:, :, slice].T); axs[2].axis('off')
    
    else:   # plot multiple slices
        mid_sagittal = image.shape[2]//2
        # plot X slices before and after the mid-sagittal slice in a grid
        fig, axs = plt.subplots(3, 14, figsize=(20, 8))
        fig.suptitle('Original Image --> Ground Truth --> Prediction')
        for i in range(14):
            axs[0, i].imshow(image[:, :, mid_sagittal-7+i].T, cmap='gray'); axs[0, i].axis('off') 
            axs[1, i].imshow(gt[:, :, mid_sagittal-7+i].T); axs[1, i].axis('off')
            axs[2, i].imshow(pred[:, :, mid_sagittal-7+i].T); axs[2, i].axis('off')

    plt.tight_layout()
    fig.show()
    return fig


class PolyLRScheduler(_LRScheduler):
    """
    Polynomial learning rate scheduler. Taken from:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/lr_scheduler/polylr.py

    """

    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


if __name__ == "__main__":

    # seed = 54
    # num_cv_folds = 10
    # names_list = FoldGenerator(seed, num_cv_folds, 100).get_fold_names()
    # tr_ix, val_tx, te_ix, fold = names_list[0]
    # print(len(tr_ix), len(val_tx), len(te_ix))

    # datalists_root = "/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/datalists/lifelong-contrast-agnostic"
    datalists_root = "/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/datalists/v2-final-aggregation-20241017"
    get_datasets_stats(datalists_root, contrasts_dict=CONTRASTS, path_save=datalists_root)
    # get_pathology_wise_split(datalists_root, path_save=datalists_root)