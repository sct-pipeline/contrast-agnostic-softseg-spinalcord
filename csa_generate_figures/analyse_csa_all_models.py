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
from scipy import stats
from charts_utils import create_experiment_folder

FNAME_LOG = 'log_stats.txt'


color_palette = {
        'nnUNet': '#fc8d62',
        'GT_soft': '#b3b3b3',
        'deepseg2d': '#e78ac3',
        'deepseg3d': '#e31a1c',
        'propseg': '#ffd92f',
        'hard_all': '#8da0cb',
        'soft_all_AdapWing': '#66c2a5',
        'soft_all': '#66c2a5',
        'soft_per_contrast': '#386cb0',
        'GT_hard': '#b3b3b3',
        'soft_all\ndice_loss': '#a6d854'
    }

#MY_PAL = {"PMJ": "cornflowerblue", "Disc": "lightyellow", "Spinal Roots":"gold"}

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
    parser.add_argument('-annotate-graph',
                        action='store_true')
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


def get_pairwise_csa(df, path_out, filename):
    fig, axs = plt.subplots(figsize=(10, 10))

    def r2(x, y, ax=None, **kws):
        ax = ax or plt.gca()
        slope, intercept, r_value, p_value, std_err = stats.linregress(x=x, y=y)
        ax.annotate(f'$r^2 = {r_value ** 2:.2f}$\nEq: ${slope:.2f}x{intercept:+.2f}$',
                    xy=(.05, .95), xycoords=ax.transAxes, fontsize=10,
                    color='darkred', backgroundcolor='#FFFFFF99', ha='left', va='top')
    def plot_diag(x, y, ax=None, **kws):
        ax = ax or plt.gca()
        ax.plot([50, 100], [50, 100], ls="--", c=".3")

    g = sns.pairplot(df, kind='reg', corner=True, plot_kws={'line_kws': {'color': 'red'}})
    g.map_lower(r2)
    # ensure axes match on each pairplot
    g.set(xlim=(55,95), ylim = (55,95))
    g.map_offdiag(plot_diag)
    plt.tight_layout()
    outfile = os.path.join(path_out, filename)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    logger.info(f'Saving {outfile}')
    plt.close()

    # Plot only T1w and T2w
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title('T1w CSA vs T2w CSA', fontsize=15, fontweight="bold")
    sns.regplot(x=df['T1w'], y=df['T2w'])
    plt.plot([50, 100], [50, 100], ls="--", c=".3")  # add diagonal line
    r2(x=df['T1w'], y=df['T2w'], ax=ax)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(55, 95)
    plt.ylim(55, 95)
    ax.set_xlabel('T1w CSA (${mm^2}$)', fontsize=14, fontweight="bold")
    ax.set_ylabel('T2w CSA (${mm^2}$)', fontsize=14, fontweight="bold")
    plt.tight_layout()
    filename_t1_t2 = 'pairplot_t1_t2.png'
    outfile = os.path.join(path_out, filename_t1_t2)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    logger.info(f'Saving {outfile}')
    plt.close()


def violin_plot(df, y_label, title, path_out, filename, set_ylim=False, annonate=False):
    sns.set_style('whitegrid', rc={'xtick.bottom': True,
                                   'ytick.left': True})
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 8))
    plt.rcParams['legend.title_fontsize'] = 16
    if "violin_plot_all" in filename :
        # NOTE: we're adding cut=0 because the STD CSA violin plot extends beyond zero
        df_filt = df.copy()
        df_filt = df_filt[(np.abs(stats.zscore(df_filt)) < 3).all(axis=1)]
        sns.violinplot(data=df_filt, ax=ax, inner="box", linewidth=2, palette=color_palette, cut=0,
                       showmeans=True, meanprops={"marker": "^", "markerfacecolor":"white", "markerscale": "3"})
        x_bot, x_top = plt.xlim()
        # overlay scatter plot on the violin plot to show individual data points
        sns.swarmplot(data=df, ax=ax, alpha=0.5, size=2, palette='dark:black')
        # Compute mean to add on plot:
        Means = df.mean()
        maximum = df_filt.max()
        STD = df.std()
        plt.scatter(x=df.columns, y=Means, c="w", marker="^", s=25, zorder=15)
        y_bot, _ = plt.ylim()
        # insert a dashed vertical line at x=1
        if 'std' in filename:
            if 'onevsall' in filename:
                x = 1.5
            else:
                x = 0.5
            ax.axvline(x=x, linestyle='--', color='black', linewidth=1, alpha=0.5)
        plt.xlim(x_bot, x_top)
        # Add mean ± std on top of each violin plot
        length = len(Means)
        for i in np.arange(len(Means)):
            textstr_F = '{:.2f} ± {:.2f}'.format(Means[i], STD[i]) + r' $mm^2$'
            yabs_max = abs(max(ax.get_ylim(), key=abs))
            y_top = yabs_max+2
            y_pos = (maximum[i])/(y_top-y_bot) + 0.1
            ax.text((i+0.5)/length, y_pos, textstr_F, transform=ax.transAxes, fontsize=15,
                    verticalalignment='top', horizontalalignment='center')

    else:
        sns.violinplot(data=df, ax=ax, inner="box", linewidth=2, palette="Set2",
                       showmeans=True, meanprops={"marker": "^", "markerfacecolor": "white", "markerscale": "3"})
        x_bot, x_top = plt.xlim()
        # overlay scatter plot on the violin plot to show individual data points
        sns.swarmplot(data=df, ax=ax, alpha=0.5, size=2, palette='dark:black')
        # Compute mean to add on plot:
        Means = df.mean()
        plt.scatter(x=df.columns, y=Means, c="w", marker="^", s=25, zorder=15)
        plt.xlim(x_bot, x_top)
        maximum = df.max()
        STD = df.std()
        # Add mean ± std on top of each violin plot
        length = len(Means)
        if 'error' in filename:
            ydiff = 15 + 2.5
            ymin = -2.5
        else:
            ydiff = 115 - 40
            ymin = 40 - 7
        if annonate:
            for i in np.arange(len(Means)):
                textstr_F = '{:.2f} ± {:.2f}'.format(Means[i], STD[i])# + r' $mm^2$'
                y_pos = (maximum[i]-ymin)/(ydiff) + 0.15
                ax.text((i+0.5)/length, y_pos, textstr_F, transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', horizontalalignment='center')

    plt.setp(ax.collections, alpha=.9)
    ax.set_title(title, pad=20, fontweight="bold", fontsize=20)
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    ax.tick_params(direction='out', axis='both')
    ax.set_ylabel(y_label, fontsize=20, fontweight="bold")
    if set_ylim:
        if 'error' in filename:
            ax.set_ylim([-2.5, 15])
        else:
            ax.set_ylim([40, 115])
            # show y-axis values in interval of 5
            ax.set_yticks(np.arange(40, 115, 5))
    else:
        yabs_max = abs(max(ax.get_ylim(), key=abs))
        ax.set_ylim(ymax=(yabs_max + 2))
    ax.xaxis.grid(True, which='minor')
    plt.tight_layout()
    outfile = os.path.join(path_out, filename)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    logger.info(f'Saving {outfile}')
    plt.close()


def main():
    exp_folder = create_experiment_folder()
    print(exp_folder)

    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(os.path.join(exp_folder, FNAME_LOG))
    logging.root.addHandler(fh)

    args = get_parser().parse_args()
    annonate = args.annotate_graph
    path_in = args.i_folder
    logger.info(path_in)
    csa_folders = os.listdir(path_in)
    logger.info(csa_folders)
    folders_included = args.include
    logger.info(f'Included: {folders_included}')
    dfs = {}
    other = [folder for folder in folders_included if 'other_methods' in folder]
    if len(other) > 0:
        other = other[0]
        folders_included.remove(other)
        folders_included.append(os.path.join(other, 'results','deepseg2d'))
        folders_included.append(os.path.join(other, 'results','deepseg3d'))
        folders_included.append(os.path.join(other, 'results','propseg'))
    print(folders_included)
    for folder in folders_included:
        print('\n', folder)
        if 'csa_gt' in folder:
            df = pd.DataFrame()
            if 'hard' in folder:
                path_csa = os.path.join(path_in, folder, 'results')  # Use binarized soft ground truth CSA results_soft_bin
            else:
                path_csa = os.path.join(path_in, folder, 'results_soft_bin')  # Use binarized soft ground truth CSA results_soft_bin
            for file in os.listdir(path_csa):
                contrast = file.split('_')
                if len(contrast) < 6:   # the filename format is "csa_soft_GT_bin_<contrast>.csv"
                    contrast = contrast[-1].split('.')[0]
                else:
                    contrast = contrast[-2]
                df[contrast] = get_csa(os.path.join(path_csa, file))
        elif 'per_contrast' in folder:
            df = pd.DataFrame()
            path_csa = os.path.join(path_in, folder)
            for file in os.listdir(path_csa):
                for file_csv in os.listdir(os.path.join(path_csa, file, 'results')):
                    if '_soft_bin.csv' in file_csv:
                        contrast = file_csv.split('_')
                        if len(contrast) < 6:   # the filename format is "csa_pred_<contrast>_soft_bin.csv"
                            contrast = contrast[-3]     # using _soft_bin.csv
                        else:
                            contrast = contrast[-4] # for mt-on and mt-off
                        df[contrast] = get_csa(os.path.join(path_csa, file, 'results', file_csv))
        elif 'csa_monai' in folder:
            df = pd.DataFrame()
            path_csa = os.path.join(path_in, folder, 'results')
            for file in os.listdir(path_csa):
                if '_soft_bin.csv' in file:     # NOTE: using binarized soft preds for plots
                    contrast = file.split('_')
                    if len(contrast) < 6:   # the filename format is "csa_pred_<contrast>_soft_bin.csv"
                        contrast = contrast[-3]     # using _soft_bin.csv
                    else:
                        contrast = contrast[-4] # for mt-on and mt-off
                    df[contrast] = get_csa(os.path.join(path_csa, file))
        else:  # For ivado
            df = pd.DataFrame()
            if 'other' in folder:
                path_csa = os.path.join(path_in, folder)
            else:
                path_csa = os.path.join(path_in, folder, 'results')
            for file in os.listdir(path_csa):
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
        'csa_nnunet_2023-08-24': 'nnUNet',
        'csa_gt_2023-08-08': 'GT_soft',
        'csa_other_methods_2023-09-21-all/results/deepseg2d': 'deepseg2d',
        'csa_other_methods_2023-09-21-all/results/deepseg3d': 'deepseg3d',
        'csa_other_methods_2023-09-21-all/results/propseg': 'propseg',
        'csa_monai_nnunet_2023-09-18': 'soft_all',
        'csa_monai_nnunet_2023-09-18_hard': 'hard_all',
        'csa_monai_nnunet_per_contrast': 'soft_per_contrast',
        'csa_gt_hard_2023-08-08': 'GT_hard',
        'csa_monai_nnunet_diceL': 'soft_all\ndice_loss'
    }

    print(dfs.keys())
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
        violin_plot(dfs[method].rename(columns={"mt-off": "GRE-T1w"}),
                    y_label=r'CSA ($\bf{mm^2}$)',
                    title="CSA across MRI contrasts " + method,
                    path_out=exp_folder,
                    filename='violin_plot_csa_percontrast_'+method+'.png',
                    set_ylim=True,
                    annonate=annonate)

        # Compute CSA error
        if method != 'GT_soft':
            error = abs(dfs[method] - dfs['GT_soft'])
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
            violin_plot(error.rename(columns={"mt-off": "GRE-T1w"}),
                        y_label=r'Absolute CSA Error ($\bf{mm^2}$)',
                        title="Absolute CSA Error across MRI contrasts " + method,
                        path_out=exp_folder,
                        filename='violin_plot_csa_percontrast_error_'+method+'.png',
                        set_ylim=True,
                        annonate=annonate)

    logger.info(f'Number of subject in test set: {len(stds.index)}')

    # Compare one model per contrast vs one for all
    include_one_vs_all = ['hard_all', 'soft_all\ndice_loss', 'soft_per_contrast', 'soft_all']
    gt = ['GT_hard', 'GT_soft']
    logger.info('\nComparing one model vs one for all')
    # Plot STD
    violin_plot(stds[gt+include_one_vs_all],
                y_label=r'Standard deviation ($\bf{mm^2}$)', 
                title="Variability of CSA across MRI contrasts",
                path_out=exp_folder,
                filename='violin_plot_all_std_onevsall.png')

    # Plot CSA error:
    violin_plot(error_csa_prediction[include_one_vs_all],
                y_label=r'Absolute CSA error ($\bf{mm^2}$)', 
                title="Absolute CSA error between prediction and GT",
                path_out=exp_folder,
                filename='violin_plot_all_csa_error_all_onevsall.png')

    # Compare MONAI_nnunet vs nnUnet vs deepseg_sc
    logger.info('\nComparing with other methods.')
    include_other_methods = ['deepseg3d', 'propseg','deepseg2d','nnUNet','soft_all']
    gt = ['GT_soft']

    # Plot STD
    violin_plot(stds[gt+include_other_methods],
                y_label=r'Standard deviation ($\bf{mm^2}$)', 
                title="Variability of CSA across contrasts\nComparison with baselines",
                path_out=exp_folder,
                filename='violin_plot_all_std_other_methods.png')

    # Plot CSA error:
    violin_plot(error_csa_prediction[include_other_methods],
                y_label=r'Absolute CSA error ($\bf{mm^2}$)',
                title="Absolute CSA error between prediction and GT\nComparison with baselines",
                path_out=exp_folder,
                filename='violin_plot_all_csa_error_all_other_methods.png')
    
    # Zoomed version
    include_other_methods = ['deepseg2d', 'nnUNet', 'soft_all']

    # Plot STD
    violin_plot(stds[gt+include_other_methods],
                y_label=r'Standard deviation ($\bf{mm^2}$)',
                title="Variability of CSA across MRI contrasts\nComparison with baselines",
                path_out=exp_folder,
                filename='violin_plot_all_std_other_methods_zoomed.png')

    # Plot CSA error:
    violin_plot(error_csa_prediction[include_other_methods],
                y_label=r'Mean absolute CSA error ($\bf{mm^2}$)',
                title="Absolute CSA error between prediction and GT\nComparison with baselines",
                path_out=exp_folder,
                filename='violin_plot_all_csa_error_all_other_methods_zoomed.png')


    # Do T1w CSA vs T2w CSA plots
    get_pairwise_csa(dfs['soft_all'], path_out=exp_folder, filename='pairwise_soft_all.png')
    # one with all 6 contrasts for the final model


if __name__ == "__main__":
    main()
