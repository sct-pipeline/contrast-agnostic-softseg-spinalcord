#!/usr/bin/env python
# -*- coding: utf-8
# Computes statistical analysis for the contrast-agnostic segmentation project
#
# For usage, type: python analyse_csa_all_models.py -h

# Authors: Sandrine Bédard

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
import itertools
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.anova import AnovaRM

FNAME_LOG = 'log_stats.txt'


color_palette = {
        'nnUNet': '#fc8d62',
        'GT_soft': '#b3b3b3',
        'DeepSeg2D': '#e78ac3',
        'deepseg3d': '#e31a1c',
        'propseg': '#ffd92f',
        'hard_all': '#8da0cb',
        'soft_all_AdapWing': '#66c2a5',
        'soft_all': '#66c2a5',
        'soft_per\ncontrast': '#386cb0',
        'GT_hard': '#b3b3b3',
        'contrast-agnostic': '#a6d854'#,
       # 'GE':'Philips', 'Siemens'}

    }


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
    parser.add_argument("-participants",
                        required=True,
                        help="Participants.tsv to get vendor information" +
                        "predictions are (for the specified contrasts).")
    parser.add_argument('-annotate-graph',
                        help='Add mean ± std annotations on the per contrasts violin plots.',
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


def get_pairwise_csa(df, df_deepseg, path_out, filename):
    """
    Creates pairwise CSA agreement and between T1w and T2w.
    Args:
        df: (pandas.DataFrame): df with CSA values of the best model.
        df_deepseg (pandas.DataFrame): df with CSA values for deepseg 2D model.
        path_out (str): Path where to save the figures.
        filename (str): Filename of pairwise plots.
    Returns:

    """
    
    fig, axs = plt.subplots(figsize=(7, 7))
    # Function to add linear equation and r^2
    def r2(x, y, ax=None, xy=(.95, .05), edgecolor='#66c2a5', **kws):
        ax = ax or plt.gca()
        slope, intercept, r_value, p_value, std_err = stats.linregress(x=x, y=y)
        ax.annotate(f'$r^2 = {r_value ** 2:.2f}$\nEq: ${slope:.2f}x{intercept:+.2f}$',
                    xy=xy, xycoords=ax.transAxes, fontsize=12,
                    color='black', backgroundcolor='#FFFFFF99', ha='right', va='bottom',
                    bbox=dict(facecolor='#FFFFFF99', alpha=0.8, edgecolor=edgecolor, boxstyle="round"))

    # Function to add identity line
    def plot_diag(x, y, ax=None, **kws):
        ax = ax or plt.gca()
        ax.plot([50, 100], [50, 100], ls="--", c=".3")
    # Create pairwise plot
    sns.set_context("paper", rc={"axes.labelsize":16,"xtick.labelsize":16, "ytick.labelsize":16})
    g = sns.pairplot(df, kind="reg", diag_kind="kde", diag_kws={"linewidth": 0, "shade": False}, corner=True, plot_kws={'line_kws': {'color': 'red'}})
    g.map_lower(r2)
    # Remove diagonal
    for i, y_var in enumerate(g.y_vars):
        for j, x_var in enumerate(g.x_vars):
            if x_var == y_var:
                g.axes[i, j].set_visible(False)
    # ensure axes match on each pairplot
    g.set(xlim=(55,95), ylim = (55,95))
    g.map_offdiag(plot_diag)
    plt.tight_layout()
    outfile = os.path.join(path_out, filename)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    logger.info(f'Saving {outfile}')
    plt.close()

    # Plot only T1w and T2w for soft_all and deepseg 2D
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title('T1w CSA vs T2w CSA', fontsize=14, fontweight="bold")
    sns.regplot(x=df['T1w'], y=df['T2w'], label='soft_all', scatter_kws={"color": "#66c2a5"}, line_kws={"color": "#66c2a5"})
    sns.regplot(x=df_deepseg['T1w'], y=df_deepseg['T2w'], label='DeepSeg2d', scatter_kws={"color": "#e78ac3"}, line_kws={"color": "#e78ac3"})
    plt.plot([50, 100], [50, 100], ls="--", c=".3")  # add diagonal line
    r2(x=df['T1w'], y=df['T2w'], ax=ax,  xy=(.95, .2))
    r2(x=df_deepseg['T1w'], y=df_deepseg['T2w'], ax=ax, edgecolor='#e78ac3')
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(55, 95)
    plt.ylim(55, 95)
    plt.legend(fontsize=11,
              loc="upper left",
              frameon=True,
              fancybox=True,
              framealpha=0.7,
              borderpad=1)
    ax.set_xlabel(r'T1w CSA ($\bf{mm^2}$)', fontsize=13, fontweight="bold")
    ax.set_ylabel(r'T2w CSA ($\bf{mm^2}$)', fontsize=13, fontweight="bold")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    filename_t1_t2 = 'pairplot_t1_t2.png'
    outfile = os.path.join(path_out, filename_t1_t2)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    logger.info(f'Saving {outfile}')
    plt.close()


def violin_plot(df, y_label, title, path_out, filename, set_ylim=False, annonate=False):
    """
    Creates violin plots overlaid with scatterplot.
    Args:
        df: (pandas.DataFrame): df.
        y_label (str): Name of y label to write on plot.
        title (str): Title of plot.
        path_out (str): Path where to save the figures.
        filename (str): Filename of pairwise plots.
        set_ylim (bool): Set y lim. Set true for per contrast plots.
        annonate (bool): Add mean ± std over violin plots for per contrast plots.
    Returns:

    """
    sns.set_style('whitegrid', rc={'xtick.bottom': True,
                                   'ytick.left': True})
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.5, 7))
    plt.rcParams['legend.title_fontsize'] = 16
    if "violin_plot_vendor" in filename:
        df_filt = df.copy()
        df_melt = df_filt.melt(id_vars='vendor', var_name='Measurement', value_name='Value')
        print(df_melt)
        group_stats = df_melt.groupby(['Measurement', 'vendor'])['Value'].agg(['mean', 'std']).reset_index()
        sns.violinplot(
            data=df_melt,
            x='Measurement',
            y='Value',
            hue='vendor',
            dodge=True,
            ax=ax,
            inner="box",
            linewidth=2,
            palette="Set2",
            width=0.8
        )
        # Overlay the swarm plot
        sns.swarmplot(
            data=df_melt,
            x='Measurement',
            y='Value',
            hue='vendor',
            dodge=True,
            ax=ax,
            alpha=0.7,
            size=3,
            palette='dark:black',
            legend=False
        )
        x_bot, x_top = plt.xlim()

        for index, row in group_stats.iterrows():
            measurement_index = list(df_melt['Measurement'].unique()).index(row['Measurement'])
            hue_index = list(df_melt['vendor'].unique()).index(row['vendor'])
            x_offset = [-0.25, 0, 0.25][hue_index]  # Adjust for hue dodge
            x_pos = measurement_index + x_offset
            y_pos = row['mean'] + row['std']  # Position above the violin
            ax.text(
                x_pos,
                y_pos+2,
                f"{row['mean']:.2f} ± {row['std']:.2f}",
                ha="center",
                va="bottom",
                fontsize=18,
                color="black"
            )

        plt.xlabel("")
        plt.xlim(x_bot, x_top)
        handles, labels = ax.get_legend_handles_labels()
        unique_handles = handles[:len(df_melt['vendor'].unique())]
        unique_labels = labels[:len(df_melt['vendor'].unique())]
        ax.legend(
            unique_handles,
            unique_labels,
            title="Vendor",
            loc="upper left",
            fontsize=16,  # Increase the font size of the legend
            title_fontsize=18  # Increase the font size of the legend title
        )

    else:
        # Ensure order of contrasts
        df = df[['DWI', 'MT-on', 'GRE-T1w', 'T1w', 'T2*w', 'T2w']]
        sns.violinplot(data=df, ax=ax, inner="box", linewidth=2, palette="Set2", width=0.9)
        x_bot, x_top = plt.xlim()
        # overlay scatter plot on the violin plot to show individual data points
        sns.swarmplot(data=df, ax=ax, alpha=0.5, size=3, palette='dark:black')
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
                ax.text((i+0.5)/length, y_pos, textstr_F, transform=ax.transAxes, fontsize=15,
                        verticalalignment='top', horizontalalignment='center')

    plt.setp(ax.collections, alpha=.9)
    ax.set_title(title, pad=20, fontweight="bold", fontsize=28)
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    plt.yticks(fontsize=21)
    plt.xticks(fontsize=21)
    ax.tick_params(direction='out', axis='both')
    ax.set_ylabel(y_label, fontsize=22, fontweight="bold")
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


def compute_paired_t_test(data):
    """
    Compute pairwise statistical test (wilcoxon) between all pairs of methods and corrects for multiple comparison.
    Args:
        data: (pandas.DataFrame): Each column represent one method.
    Returns:

    """
    columns = list(data.columns)
    combinations = list(itertools.combinations(columns, 2))
    logger.info(combinations)
    # Run Wilcoxon (groups are dependent, same subjects)
    pvals = []
    for i in range(len(combinations)):
        stat, pval = stats.wilcoxon(x=data[combinations[i][0]], y=data[combinations[i][1]])
        pvals.append(pval)
    p_adjusted = multipletests(pvals, method='bonferroni', alpha=0.05)
    logger.info(p_adjusted)


def compute_anova(df, depvar='std', subject='Subject', within=None, aggregate_func=None):
    results = AnovaRM(data=df, depvar=depvar, subject=subject, within=within, aggregate_func=aggregate_func).fit()
    p_value = results.anova_table["Pr > F"][0]
    logger.info('ANOVA: {}'.format(results))
    logger.info('ANOVA p-value: {}'.format(p_value))


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
    participants = args.participants
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
    # Read CSV files
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
            for file in os.listdir(path_csa)[::-1]:
                if 'csv' in file:
                    contrast = file.split('_')
                    if len(contrast) < 4:   # the filename format is "csa_pred_<contrast>.csv"
                        contrast = contrast[-1].split('.')[0]
                    else:
                        contrast = contrast[-2]
                    series_contrast = get_csa(os.path.join(path_csa, file)).rename(columns={'MEAN(area)': contrast})
                    series_contrast = series_contrast[~series_contrast.index.duplicated(keep='last')]
                    df = pd.concat([df, series_contrast], axis=1)
                    #df[contrast] = get_csa(os.path.join(path_csa, file))
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
        'contrast-agn': 'contrast-agnostic',
        'deepseg': 'DeepSeg2D',
    }

    print(dfs.keys())
    dfs = dict((rename[key], value) for (key, value) in dfs.items())
    stds = pd.DataFrame()
    error_csa_prediction = pd.DataFrame()
    # Compute CSA STD and CSA error for all methods
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
        violin_plot(dfs[method].rename(columns={"mt-off": "GRE-T1w", "T2star": "T2*w", "mt-on": "MT-on", "dwi":"DWI"}),
                    y_label=r'CSA ($\bf{mm^2}$)',
                    title="CSA across MRI contrasts " + method,
                    path_out=exp_folder,
                    filename='violin_plot_csa_percontrast_'+method+'.png',
                    set_ylim=True,
                    annonate=annonate)

        # Compute CSA error
        # if method != 'GT_soft':
        #     error = abs(dfs[method] - dfs['GT_soft'])
        #     logger.info('Max error:')
        #     logger.info(error.max())
        #     logger.info('Index of max')
        #     logger.info(error.idxmax())
        #     # Append all errors in one column for CSA error across methods
        #     oneCol = []
        #     for column in error.columns:
        #         oneCol.append(error[column])
        #     error_csa_prediction[method] = pd.concat(oneCol, ignore_index=True)
        #     # Plot error per contrast
        #     violin_plot(error.rename(columns={"mt-off": "GRE-T1w", "T2star": "T2*w", "mt-on": "MT-on", "dwi":"DWI"}),
        #                 y_label=r'Absolute CSA Error ($\bf{mm^2}$)',
        #                 title="Absolute CSA Error across MRI contrasts " + method,
        #                 path_out=exp_folder,
        #                 filename='violin_plot_csa_percontrast_error_'+method+'.png',
        #                 set_ylim=True,
        #                 annonate=annonate)

    logger.info(f'Number of subject in test set: {len(stds.index)}')
    # Add participants.tsv information: 
    df_participants = pd.read_csv(participants, sep='\t').set_index('participant_id')
    stds["vendor"] = df_participants["manufacturer"]
    print(stds)
    violin_plot(stds, y_label=r'Standard Deviation of CSA ($\bf{mm^2}$)',
                title="Variability of CSA across MRI contrast",
                path_out=exp_folder,
                filename="violin_plot_vendor.png",
                set_ylim=False,
                annonate=annonate)
    
    # Compute statistical test for soft_all CSA
    ###########################################
    logger.info('\nPaired ANOVA test between CSA across contrasts for soft_all')
    csa_soft_all = dfs['contrast-agnostic'].copy().reset_index()
    print(csa_soft_all.columns)
    length = len(csa_soft_all.index.to_list())
    print('Lenght:', length)
    #print(csa_soft_all)
    #csa_soft_all['csa'] = csa_soft_all['DWI']
    #csa_soft_all['contrast'] = 'DWI'
    oneCol = []
    subjects = []
    cols = ['dwi', 'mt-on', 'mt-off', 'T1w', 'T2star', 'T2w']
    for col in cols:
        oneCol.append(csa_soft_all[col])
        subjects.append(csa_soft_all['Subject'])
    csa_row = pd.concat(oneCol, ignore_index=True)
    csa = pd.DataFrame()
    csa['csa'] = csa_row
    csa['Subject'] = np.ravel(subjects)
    csa['contrast'] = 'test'
    csa.loc[0:length, 'contrast'] = 'DWI'
    csa.loc[length:2*length, 'contrast'] = 'MT-on'
    csa.loc[2*length:3*length, 'contrast'] = 'GRE-T1w'
    csa.loc[3*length:4*length, 'contrast'] = 'T1w'
    csa.loc[4*length:5*length, 'contrast'] = 'T2*w'
    csa.loc[5*length:6*length, 'contrast'] = 'T2w'
    print(csa)
    compute_anova(df=csa, depvar='csa', subject='Subject', within=['contrast'])


    # Do T1w CSA vs T2w CSA plots
    #################################
    get_pairwise_csa(dfs['contrast-agnostic'].rename(columns={"mt-off": "GRE-T1w", "T2star": "T2*w", "mt-on": "MT-on", "dwi":"DWI"}), dfs['DeepSeg2D'], path_out=exp_folder, filename='pairwise_soft_all.png')

    # TODO: per vendor

if __name__ == "__main__":
    main()
