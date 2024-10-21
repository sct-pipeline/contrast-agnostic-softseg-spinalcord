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
    

def plot_contrast_wise_pathology(df, path_save):
    # remove the TOTAL row
    df = df[:-1]
    # remove the #total_per_contrast column
    df = df.drop(columns=['#total_per_contrast'])

    color_palette = {
        'HC': '#fc8d62',
        'MS': '#b3b3b3',
        'RIS': '#e78ac3',
        'RRMS': '#e31a1c',
        'PPMS': '#ffd92f',
        'DCM': '#8da0cb',
        'SPMS': '#66c2a5',
        'SCI': '#66c2a5',
        'NMO': '#386cb0',
        'SYR': '#b3b3b3',
        'ALS': '#a6d854',
        'LBP': '#cab2d6'
    }

    contrasts = df.index.tolist()

    # plot a pie chart for each contrast and save as different file
    for contrast in contrasts:
        df_contrast = df.loc[[contrast]].T
        # reorder the columsn to put 'ALS' between 'HC' and 'MS'
        if contrast in ['dwi']:
            df_contrast = df_contrast.reindex(['ALS', 'HC', 'MS', 'DCM', 'SCI', 'NMO', 'RRMS', 'PPMS', 'SPMS', 'RIS', 'LBP', 'SYR'])
        elif contrast in ['unit1']:
            # reorder the columsn to put 'PPMS' between 'MS' and 'RRMS'
            df_contrast = df_contrast.reindex(['HC', 'MS', 'PPMS', 'RRMS', 'SPMS', 'RIS', 'DCM', 'SCI', 'NMO', 'ALS', 'LBP', 'SYR'])
        elif contrast in ['t2star']:
            df_contrast = df_contrast.reindex(['HC', 'ALS', 'MS', 'DCM', 'SCI', 'NMO', 'RRMS', 'PPMS', 'SPMS', 'RIS', 'LBP', 'SYR'])

        df_contrast = df_contrast[df_contrast[contrast] != 0]
        
        fig, ax = plt.subplots(figsize=(5.5, 3.5), subplot_kw=dict(aspect="equal"))  # Increased figure size
        wedges, texts = ax.pie(
            df_contrast[contrast], 
            wedgeprops=dict(width=0.5), 
            startangle=-40,
            colors=[color_palette[pathology] for pathology in df_contrast.index],
        )

        # Annotation customization
        bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
        texts_to_adjust = []  # collect all annotations for adjustment

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            # font size
            kw["fontsize"] = 11
            # bold font
            kw["fontweight"] = 'bold'

            # Skip annotation for 'SYR'
            if df_contrast.index[i] == 'SYR':
                continue

            # Push small labels further away from pie
            distance = 1.1 #1.4 if df_contrast[contrast].iloc[i] / df_contrast[contrast].sum() < 0.1 else 1.2
            # for dwi contrast and sci pathology, plot the annotation to the left
            if contrast == 'dwi' and df_contrast.index[i] == 'SCI':
                distance = 1.4
                horizontalalignment = 'left'
            # plot 'ALS' annotation to the right
            if df_contrast.index[i] == 'ALS':
                distance = 1.2
                horizontalalignment = 'left'                
            if contrast == 't2w' and df_contrast.index[i] == 'RIS':
                distance = 1.4
                horizontalalignment = 'left'
                
            # Annotate with number of images per pathology
            text = f"{df_contrast.index[i]} (n={df_contrast.iloc[i, 0]})"
            annotation = ax.annotate(text, xy=(x, y), xytext=(distance*np.sign(x), distance*y),
                                     horizontalalignment=horizontalalignment, **kw)
            texts_to_adjust.append(annotation)

        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(path_save, f'{contrast}_pathology_split.png'), dpi=300)
        plt.close()



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


    # get the contrast-wise pathology split
    df_contrast_pathology = pd.DataFrame.from_dict(pathology_contrasts, orient='index')
    # sort the dataframe by the contrast column
    df_contrast_pathology = df_contrast_pathology.sort_index()
    # add a row for the total number of images
    df_contrast_pathology.loc['TOTAL'] = df_contrast_pathology.sum()
    # add a column for the total number of images per contrast
    df_contrast_pathology['#total_per_contrast'] = df_contrast_pathology.sum(axis=1)
    # print(df_contrast_pathology)
    
    # plots 
    save_path = os.path.join(path_save, 'plots')
    os.makedirs(save_path, exist_ok=True)
    plot_contrast_wise_pathology(df_contrast_pathology, save_path)
    # exit()

    # create a txt file
    with open(os.path.join(path_save, 'dataset_stats_overall.txt'), 'w') as f:
        # 1. write the datalists used in a bullet list
        f.write(f"DATASETS USED FOR MODEL TRAINING (n={len(csvs)}):\n")
        for csv in csvs:
            # only write the dataset name
            f.write(f"\t- {csv.split('_')[1]}\n")
        f.write("\n")

        # 2. write the subject-wise pathology split
        f.write(f"\nSUBJECT-WISE PATHOLOGY SPLIT:\n\n")
        f.write(df_pathology.to_markdown())
        f.write("\n\n\n")

        # 3. write the contrast-wise pathology split (a subject can have multiple contrasts)
        f.write(f"CONTRAST-WISE PATHOLOGY SPLIT (a subject can have multiple contrasts):\n\n")
        f.write(df_contrast_pathology.to_markdown())
        f.write("\n\n\n")

        # 4. write the train/validation/test split per contrast
        f.write(f"SPLITS ACROSS DIFFERENT CONTRASTS (n={len(contrasts_final)}):\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")



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