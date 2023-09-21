#!/usr/bin/env python
# -*- coding: utf-8


import os
import logging
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from charts_utils import create_experiment_folder

FNAME_LOG = 'log_stats.txt'

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i-folder",
                        required=True,
                        help="Folder in which the CSA values for the segmentation" +
                        "predictions are (for the specified contrasts).")
    parser.add_argument("-include",
                        required=True,
                        nargs='+',
                        default=[],
                        help="Folder names to include" +
                        "predictions are (for the specified contrasts).")

    return parser


def get_csa(csa_filename):
    """
    From .csv output file of process_data.sh (sct_process_segmentation),
    returns a panda dataFrame with CSA values sorted by subject eid.
    Args:
        csa_filename (str): filename of the .csv file that contains de CSA values
    Returns:
        csa (pd.Series): column of CSA values

    """
    sc_data = pd.read_csv(csa_filename)
    csa = pd.DataFrame(sc_data[['Filename', 'MEAN(area)']]).rename(columns={'Filename': 'Subject'})
    # Add a columns with subjects eid from Filename column
    csa.loc[:, 'Subject'] = csa['Subject'].str.split('/').str[-3]   #sub-barcelona05-dwi_018.nii.gz  sub-beijingGE03_rec-average_dwi_pred_T0000.nii.gz
    # Set index to subject eid
    csa = csa.set_index('Subject')
    return csa


def violin_plot(df, y_label, title, path_out, filename, set_ylim=False):
    sns.set_style('whitegrid', rc={'xtick.bottom': True,
                                   'ytick.left': True})
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 10))
    plt.rcParams['legend.title_fontsize'] = 'x-large'
    plt.yticks(fontsize="x-large")
    
    if filename == 'violin_plot_all.png':
        # NOTE: we're adding cut=0 because the STD CSA violin plot extends beyond zero
        sns.violinplot(data=df, ax=ax, inner="box", linewidth=2, palette="Set2", cut=0,
                       showmeans=True, meanprops={"marker": "^", "markerfacecolor":"white", "markerscale": "2"})
        x_bot, x_top = plt.xlim()
        # overlay scatter plot on the violin plot to show individual data points
        sns.swarmplot(data=df, ax=ax, alpha=0.5, size=3, palette='dark:black')
        # insert a dashed vertical line at x=5
        ax.axvline(x=4.5, linestyle='--', color='black', linewidth=1, alpha=0.5)
        plt.xlim(x_bot, x_top)
    else:
        sns.violinplot(data=df, ax=ax, inner="box", linewidth=2, palette="Set2",
                       showmeans=True, meanprops={"marker":"^", "markerfacecolor":"white", "markerscale": "2"})
        x_bot, x_top = plt.xlim()
        # # overlay scatter plot on the violin plot to show individual data points
        sns.swarmplot(data=df, ax=ax, alpha=0.5, size=3, palette='dark:black')
        # Compute mean to add on plot:
        Means = df.mean()
        plt.scatter(x=df.columns, y=Means, c="w", marker="^", s=20, zorder=15)
        plt.xlim(x_bot, x_top)

    plt.setp(ax.collections, alpha=.9)
    ax.set_title(title, pad=20, fontweight="bold", fontsize=17)
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.grid(True, which='minor')
    #ax.set_xticks(np.arange(0, len(labels)), labels, rotation=30, ha='right', fontsize=17)
    plt.yticks(fontsize=17)
    plt.xticks(fontsize=15)
    ax.tick_params(direction='out', axis='both')
    #ax.set_xlabel('Segmentation type', fontsize=17, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=17, fontweight="bold")
    if set_ylim:
        if 'error' in filename:
            ax.set_ylim([-2.5, 14])
        else:
            ax.set_ylim([40, 105])
            # show y-axis values in interval of 5
            ax.set_yticks(np.arange(40, 105, 5))
            # insert horizontal line at y=70 
            ax.axhline(y=70, linestyle='--', color='black', linewidth=1.5, alpha=0.5)

    else:
        yabs_max = abs(max(ax.get_ylim(), key=abs))
        ax.set_ylim(ymax=(yabs_max + 2))

    plt.tight_layout()
    outfile = os.path.join(path_out, filename)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")

def main():
    exp_folder = create_experiment_folder()
    print(exp_folder)

    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(os.path.join(exp_folder, FNAME_LOG))
    logging.root.addHandler(fh)


    args = get_parser().parse_args()
    path_in = args.i_folder
    logger.info(path_in)
    csa_folders = os.listdir(path_in)
    logger.info(csa_folders)
    folders_included = args.include
    logger.info(f'Included: {folders_included}')
    dfs = {}
    for folder in csa_folders:
        if folder in folders_included:
            if 'csa_gt' in folder:
                df = pd.DataFrame()
                path_csa = os.path.join(path_in, folder, 'results_soft_bin')  # Use binarized soft ground truth CSA results_soft_bin
                # path_csa = os.path.join(path_in, folder)    # TODO: comment this when NOT comparing GTs only
                for file in os.listdir(path_csa):
                    if 'csa_soft_GT_bin' in file:
                        contrast = file.split('_')
                        if len(contrast) < 6:   # the filename format is "csa_soft_GT_bin_<contrast>.csv"
                            contrast = contrast[-1].split('.')[0]
                            print(contrast)
                        else:
                            contrast = contrast[-2]
                            print(contrast)
                        df[contrast] = get_csa(os.path.join(path_csa, file))

            elif 'csa_monai' in folder:
                df = pd.DataFrame()
                path_csa = os.path.join(path_in, folder, 'results')
                for file in os.listdir(path_csa):
                    if '_soft_bin.csv' in file:     # NOTE: using binarized soft preds for plots
                        print(file)
                        contrast = file.split('_')
                        if len(contrast) < 6:   # the filename format is "csa_pred_<contrast>_soft_bin.csv"
                            contrast = contrast[-3]     # using _soft_bin.csv
                        else:
                            contrast = contrast[-4] # for mt-on and mt-off
                        df[contrast] = get_csa(os.path.join(path_csa, file))

            else:  # For ivado
                df = pd.DataFrame()
                path_csa = os.path.join(path_in, folder, 'results')
                for file in os.listdir(path_csa):
                    print(file)
                    if 'csv' in file:
                        contrast = file.split('_')
                        if len(contrast) < 4:   # the filename format is "csa_pred_<contrast>.csv"
                            contrast = contrast[-1].split('.')[0]
                        else:
                            contrast = contrast[-2]
                        df[contrast] = get_csa(os.path.join(path_csa, file))
            dfs[folder] = df
            df.dropna(axis=0, inplace=True)

    # only retain the common subjects from each folder inside dfs
    # NOTE: only requirement is that the first folder in folders_included should NOT be the GT folder
    common_subjects = dfs[folders_included[0]].index
    for folder in folders_included[1:]:
        common_subjects = common_subjects.intersection(dfs[folder].index)
    for folder in folders_included:
        dfs[folder] = dfs[folder].loc[common_subjects]
        dfs[folder] = dfs[folder].sort_index()
        logger.info(f'Number of subjects in {folder}: {len(dfs[folder].index)}')

    # To compare only monai model and the GT
    rename = {
        'ivado_soft_no_crop':'IVADO_avg_bin',
        #'ivado_avg_bin_no_crop': 'IVADO_avg_train_bin', #--> binarized before training
        'csa_nnunet_2023-08-24': 'nnUNet_avg_bin',
        'csa_monai_unet_bestValCSA': 'MONAI_UNet_avg',
        # 'csa_monai_unetr_bestValCSA': 'MONAI_UNetR_avg_bin',       # Added folder name here
        'csa_monai_nnunet_2023-09-04': 'MONAI_nnUnet',
        'csa_monai_ivado_reImp': 'csa_monai_ivado_reImp',
        'csa_gt_2023-08-08': 'GT_soft_avg_bin'
    }


    dfs = dict((rename[key], value) for (key, value) in dfs.items())
    stds = pd.DataFrame()
    error_csa_prediction = pd.DataFrame()
    for method in dfs.keys():
        std = dfs[method].std(axis=1)
        mean_std = std.mean()
        std_std = std.std()
        logger.info(f'\nProcessing {method}')
        logger.info(f'{method} Mean STD: {mean_std} ± {std_std} mm^2')
        logger.info(f'{method} Mean CSA: {dfs[method].values.mean()} ± {dfs[method].values.std(ddof=1)} mm^2')
        logger.info(f'{method} Mean CSA: \n{dfs[method].mean()}')
        logger.info(f'{method} STD CSA: \n{dfs[method].std()}')
        logger.info(std.sort_values(ascending=False))
        stds[method] = std
        violin_plot(dfs[method],
                    y_label=r'CSA ($\bf{mm^2}$)',
                    title="CSA across MRI contrasts " + method,
                    path_out=exp_folder,
                    filename='violin_plot_csa_percontrast_'+method+'.png',
                    set_ylim=True)

        # Compute CSA error
        if method != 'GT_soft_avg_bin':
            error = abs(dfs[method] - dfs['GT_soft_avg_bin'])
            logger.info('Max error:')
            logger.info(error.max())
            logger.info('Index of max')
            logger.info(error.idxmax())
            # Append all errors in one column for CSA error across methods
            oneCol = []
            for column in error.columns:
                oneCol.append(error[column])
            error_csa_prediction[method] = pd.concat(oneCol, ignore_index=True)
            # Plot error per contrast
            violin_plot(error,
                        y_label=r'Absolute CSA Error($\bf{mm^2}$)',
                        title="CSA Error across MRI contrasts " + method,
                        path_out=exp_folder,
                        filename='violin_plot_csa_percontrast_error_'+method+'.png',
                        set_ylim=True)

    logger.info(f'Number of subject in test set: {len(stds.index)}')
    # Plot STD
    violin_plot(stds,
                y_label=r'Standard deviation ($\bf{mm^2}$)', 
                title="Variability of CSA across MRI contrasts",
                path_out=exp_folder,
                filename='violin_plot_all.png')

    # Plot CSA error:
    violin_plot(error_csa_prediction,
                y_label=r'Mean absolute CSA error($\bf{mm^2}$)', 
                title="Absolute CSA error between prediction and GT",
                path_out=exp_folder,
                filename='violin_plot_csa_error_all.png')

if __name__ == "__main__":
    main()
