#!/usr/bin/env python
# -*- coding: utf-8


import os
import logging
import argparse
import sys
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
                                 'ytick.left': True,})
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 9)) 
    plt.rcParams['legend.title_fontsize'] = 'x-large'
    plt.yticks(fontsize="x-large")
    sns.violinplot(data=df, ax=ax, inner="box", linewidth=2, palette="Set2")
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
        ax.set_ylim([40, 105])
    else:
        yabs_max = abs(max(ax.get_ylim(), key=abs))
        ax.set_ylim(ymax=(yabs_max + 1))

    plt.tight_layout()
    outfile = os.path.join(path_out, filename)
    plt.savefig(outfile, dpi=400, bbox_inches="tight")

def main():
    exp_folder = create_experiment_folder()

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
            df = pd.DataFrame()
            path_csa = os.path.join(path_in, folder, 'results')
            for file in os.listdir(path_csa):
                if 'csv' in file:
                    contrast = file.split('_')
                    if len(contrast) < 4:
                        contrast = contrast[-1].split('.')[0]
                    else:
                        contrast = contrast[-2]
                    df[contrast] = get_csa(os.path.join(path_csa, file))
            dfs[folder] = df
            df.dropna(axis=0, inplace=True)
    # Compute STD

    rename = {'ivado_avg_bin_no_crop': 'IVADO_avg_bin',
             'ivado_soft_no_crop':'IVADO_avg',
             'ivado_hard_GT': 'IVADO_hard_GT',
             'csa_nnunet_soft_avg_all_no_crop': 'nnUnet_avg_bin',
             }
    dfs = dict((rename[key], value) for (key, value) in dfs.items())
    stds = pd.DataFrame()
    for method in dfs.keys():
        std = dfs[method].std(axis=1)
        mean_std = std.mean()
        std_std = std.std()
        logger.info(f'{method} Mean STD: {mean_std} ± {std_std} mm^2')
        logger.info(f'{method} Mean CSA: {dfs[method].values.mean()} ± {dfs[method].values.std(ddof=1)} mm^2')
        logger.info(f'{method} Mean CSA: \n{dfs[method].mean()}')
        logger.info(f'{method} STD CSA: \n{dfs[method].std()}')
        stds[method] = std
        violin_plot(dfs[method], 
                    y_label=r'CSA ($\bf{mm^2}$)',
                    title="CSA across MRI contrasts " + method,
                    path_out=exp_folder, 
                    filename='violin_plot_csa_percontrast_'+method+'.png', 
                    set_ylim=True)
    logger.info(f'Number of subject in test set: {len(stds.index)}')
    stds = stds[['IVADO_hard_GT', 'IVADO_avg', 'IVADO_avg_bin', 'nnUnet_avg_bin']]
    
    violin_plot(stds,
                y_label=r'Standard deviation ($\bf{mm^2}$)', 
                title="Variability of CSA across MRI contrasts",
                path_out=exp_folder,
                filename='violin_plot_all.png')



if __name__ == "__main__":
    main()
