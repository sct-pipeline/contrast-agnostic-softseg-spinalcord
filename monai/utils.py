import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import torch
import subprocess
import os
import json
import pandas as pd
import re
import importlib
import pkgutil
from batchgenerators.utilities.file_and_folder_operations import *


CONTRASTS = {
    "t1map": ["T1map"],
    "mp2rage": ["inv-1_part-mag_MP2RAGE", "inv-2_part-mag_MP2RAGE"],
    "t1w": ["T1w", "space-other_T1w"],
    "t2w": ["T2w", "space-other_T2w"],
    "t2star": ["T2star", "space-other_T2star"],
    "dwi": ["rec-average_dwi", "acq-dwiMean_dwi"],
    "mt-on": ["flip-1_mt-on_space-other_MTS", "acq-MTon_MTR"],
    "mt-off": ["flip-2_mt-off_space-other_MTS"],
    "unit1": ["UNIT1"],
    "psir": ["PSIR"],
    "stir": ["STIR"]
}

def get_datasets_stats(datalists_root, contrasts_dict, path_save):

    datalists = [file for file in os.listdir(datalists_root) if file.endswith('_seed50.json')]

    df = pd.DataFrame(columns=['train', 'validation', 'test'])
    # collect all the contrasts from the datalists
    for datalist in datalists:
        json_path = os.path.join(datalists_root, datalist)
        with open(json_path, 'r') as f:
            data = json.load(f)
            data = data['numImagesPerContrast'].items()

        for split, contrast_info in data:
            for contrast, num_images in contrast_info.items():
                # add the contrast and the number of images to the dataframe
                if contrast not in df.index:
                    df.loc[contrast] = [0, 0, 0]
                df.loc[contrast, split] += num_images

    # reshape dataframe and add a column for contrast
    df = df.reset_index()
    df = df.rename(columns={'index': 'contrast'})

    contrasts_final = list(contrasts_dict.keys())

    # rename the contrasts column as per contrasts_final
    for c in df['contrast'].unique():
        for cf in contrasts_final:
            if re.search(cf, c.lower()):
                df.loc[df['contrast'] == c, 'contrast'] = cf
                break
    
    # NOTE: MTon-MTR is same as flip-1_mt-on_space-other_MTS, but the naming is not mt-on
    # so doing the renaming manually
    df.loc[df['contrast'] == 'acq-MTon_MTR', 'contrast'] = 'mt-on'

    # sum the duplicate contrasts 
    df = df.groupby('contrast').sum().reset_index()
    
    # rename columns
    df = df.rename(columns={'contrast': 'Contrast', 'train': '#train_images', 'validation': '#validation_images', 'test': '#test_images'})
    # add a row for the total number of images
    df.loc[len(df)] = ['TOTAL', df['#train_images'].sum(), df['#validation_images'].sum(), df['#test_images'].sum()]
    # print(df.to_markdown(index=False))
    df.to_markdown(os.path.join(path_save, 'dataset_split.md'), index=False)

    # # total number of images for each split
    # print(f"Total number of training images: {df['#train_images'].sum()}")
    # print(f"Total number of validation images: {df['#validation_images'].sum()}")
    # print(f"Total number of test images: {df['#test_images'].sum()}")

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


def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:

    * FP = Soft False Positives
    * FN = Soft False Negatives
    * TP = Soft True Positives
    * TN = Soft True Negatives

    Robust to hard or soft input masks. For example::
        prediction=np.asarray([0, 0.5, 1])
        groundtruth=np.asarray([0, 1, 1])
        Leads to FP = 1.5

    Note: It assumes input values are between 0 and 1.

    Args:
        prediction (ndarray): Binary prediction.
        groundtruth (ndarray): Binary groundtruth.

    Returns:
        float, float, float, float: FP, FN, TP, TN
    """
    FP = float(np.sum(prediction * (1.0 - groundtruth)))
    FN = float(np.sum((1.0 - prediction) * groundtruth))
    TP = float(np.sum(prediction * groundtruth))
    TN = float(np.sum((1.0 - prediction) * (1.0 - groundtruth)))
    return FP, FN, TP, TN


def precision_score(prediction, groundtruth, err_value=0.0):
    """Positive predictive value (PPV).

    Precision equals the number of true positive voxels divided by the sum of true and false positive voxels.
    True and false positives are computed on soft masks, see ``"numeric_score"``.
    Taken from: https://github.com/ivadomed/ivadomed/blob/master/ivadomed/metrics.py

    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
        err_value (float): Value returned in case of error.

    Returns:
        float: Precision score.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP) <= 0.0:
        return err_value

    precision = np.divide(TP, TP + FP)
    return precision


def recall_score(prediction, groundtruth, err_value=0.0):
    """True positive rate (TPR).

    Recall equals the number of true positive voxels divided by the sum of true positive and false negative voxels.
    True positive and false negative values are computed on soft masks, see ``"numeric_score"``.
    Taken from: https://github.com/ivadomed/ivadomed/blob/master/ivadomed/metrics.py

    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
        err_value (float): Value returned in case of error.

    Returns:
        float: Recall score.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return err_value
    TPR = np.divide(TP, TP + FN)
    return TPR


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


def compute_average_csa(patch, spacing):
    num_slices = patch.shape[2]
    areas = torch.empty(num_slices)
    for slice_idx in range(num_slices):
        slice_mask = patch[:, :, slice_idx]
        area = torch.count_nonzero(slice_mask) * (spacing[0] * spacing[1])
        areas[slice_idx] = area

    return torch.mean(areas)


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
    datalists_root = "/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/datalists/aggregation-20240517"
    get_datasets_stats(datalists_root, contrasts_dict=CONTRASTS, path_save=datalists_root)