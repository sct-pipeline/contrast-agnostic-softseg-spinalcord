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
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i-folder",
                        required=True,
                        help="Folder in which the CSA values for the segmentation" +
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


def violin_plot(df, y_label, path_out, filename):
    sns.set_style('whitegrid', rc={'xtick.bottom': True,
                                 'ytick.left': True,})
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 9)) 
    plt.rcParams['legend.title_fontsize'] = 'x-large'
    plt.yticks(fontsize="x-large")
    sns.violinplot(data=df, ax=ax, inner="box", linewidth=2, palette="Set2")
    plt.setp(ax.collections, alpha=.9)
    ax.set_title("Variability of CSA across MRI contrasts", pad=20, fontweight="bold", fontsize=17)
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.grid(True, which='minor')
    #ax.set_xticks(np.arange(0, len(labels)), labels, rotation=30, ha='right', fontsize=17)
    plt.yticks(fontsize=17)
    ax.tick_params(direction='out', axis='both')
    #ax.set_xlabel('Segmentation type', fontsize=17, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=17, fontweight="bold")
    yabs_max = abs(max(ax.get_ylim(), key=abs))
    ax.set_ylim(ymax=(yabs_max + 1))

    plt.tight_layout()
    outfile = os.path.join(path_out, filename)
    plt.savefig(outfile, dpi=400, bbox_inches="tight")

def main():
    exp_folder = create_experiment_folder()

    args = get_parser().parse_args()
    path_in = args.i_folder
    print(path_in)
    csa_folders = os.listdir(path_in)
    print(csa_folders)
    dfs = {}
    for folder in csa_folders:
        df = pd.DataFrame()
        path_csa = os.path.join(path_in, folder, 'results')
        for file in os.listdir(path_csa):
            print(file)
            if 'csv' in file:
                contrast = file.split('_')
                if len(contrast) < 4:
                    contrast = contrast[-1].split('.')[0]
                else:
                    contrast = contrast[-2]
                print(contrast)
                df[contrast] = get_csa(os.path.join(path_csa, file))
               # print(dataset)
        dfs[folder] = df
    print(df)

    # Drop nan
    df.dropna(axis=0, inplace=True)
    # Compute STD
    std = df.std(axis=1)
    print('Mean STD:', std.mean())
   # print(std)
    violin_plot(df, r'CSA ($\bf{mm^2}$)', exp_folder, 'violin_plot_csa_percontrast.png')
    violin_plot(std, r'Standard deviation ($\bf{mm^2}$)', exp_folder, 'violin_plot_all.png')


if __name__ == "__main__":
    main()
