"""
Generate violin plots from CSV data across different models (methods), thresholds, and resolutions.

Usage:
    python generate_figure_csa_across_resolutions.py -i /path/to/data.csv -a [methods, resolutions, thresholds]
"""

import os
import argparse
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Setting the hue order as specified
HUE_ORDER = ["softseg_bin", "deepseg", "plain_320", "plain_384", "resencM"]
# HUE_ORDER = ["softseg_bin", "deepseg_2d", "monai_single", "monai_7datasets", "swinunetr_7datasets"]
# HUE_ORDER = ["softseg_bin", "monai_v21", "monai_v23", "monai_v2x"]
HUE_ORDER_RES = ["1mm", "05mm", "15mm", "3mm", "2mm"]
CONTRAST_ORDER = ["DWI", "MTon", "MToff", "T1w", "T2star", "T2w"]

FONTSIZE = 12
XTICKS = ["GT", "DeepSeg2D", "C-A\nplain_320", "C-A\nplain_384", "C-A\nresencM"]
# XTICKS = ["GT", "contrast-agnostic\nv2.1", "contrast-agnostic\nv2.3", "contrast-agnostic\nv2.x"] 


def save_figure(file_path, save_fname):
    plt.tight_layout()
    save_path = os.path.join(file_path, save_fname)
    plt.savefig(save_path, dpi=300)
    print(f'Figure saved to {save_path}')


def fetch_participant_id(filename_path):
    """
    Get participant_id from the input BIDS-compatible filename or file path
    :return: participant_id: subject ID (e.g., sub-001)
    """

    _, filename = os.path.split(filename_path)              # Get just the filename (i.e., remove the path)
    participant = re.search('sub-(.*?)[_/]', filename_path)     # [_/] slash or underscore
    participant_id = participant.group(0)[:-1] if participant else ""    # [:-1] removes the last underscore or slash
    # REGEX explanation
    # \d - digit
    # \d? - no or one occurrence of digit
    # *? - match the previous element as few times as possible (zero or more times)
    # . - any character

    return participant_id


# Function to extract contrast and method from the filename
def extract_contrast_and_details(filename, across="Method"):
    """
    Extract the segmentation method and resolution from the filename.
    The method (e.g., propseg, deepseg_2d, nnunet_3d_fullres, monai) and resolution (e.g., 1mm)
    are embedded in the filename.
    """
    # pattern = r'.*iso-(\d+mm).*_(propseg|deepseg_2d|nnunet_3d_fullres|monai).*'
    if across == "Method":
        # pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg_bin|deepseg_2d|soft_input|bin_input).*'
        pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg_bin|deepseg|plain_320|plain_384|resencM).*'
        # pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg_bin|deepseg_2d|monai_single|monai_7datasets|swinunetr_7datasets).*'
        # pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg_bin|monai_v21|monai_v23|monai_v2x).*'
        match = re.search(pattern, filename)
        if match:
            return match.group(1), match.group(2)
        else:
            return 'Unknown', 'Unknown'

    elif across == "Resolution":
        pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w)_res_(\d+mm).*_(deepseg_2d|nnunet|monai|swinunetr|mednext).*'
        match = re.search(pattern, filename)
        if match:
            return match.group(1), match.group(3), match.group(2)
        else:
            return 'Unknown', 'Unknown', 'Unknown'

    else:
        raise ValueError(f'Unknown analysis type: {across}. Choices: [Method, Resolution, Threshold].')
    


def generate_figure_std(data, file_path, across="Method", metric="csa", hue_order=HUE_ORDER):
    """
    Generate violinplot showing STD across participants for each method
    """
    if across == "Threshold":
        # create a dataframe with only "monai"
        data = data[data['Method'] == "monai"]
    elif across == "Resolution":
        # create a dataframe with only the specified model
        model = "monai"
        data = data[data['Method'] == model]

    if metric == "csa":
        # Compute mean and std across contrasts for each method
        df = data.groupby([across, 'Participant'])['MEAN(area)'].agg(['mean', 'std']).reset_index()
    elif metric == "swdice":
        # Compute mean and std across contrasts for each method
        df = data.groupby([across, 'Participant'])['average_slicewise_dice'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(12, 6))
    sns.violinplot(x=across, y='std', data=df, order=hue_order)
    # overlay swarm plot on the violin plot to show individual data points
    sns.swarmplot(x=across, y='std', data=df, color='k', order=hue_order, size=3)
    # plt.xticks(rotation=45)
    plt.xlabel(across)
    if metric == "csa":
        plt.ylabel('STD [mm^2]')
        plt.title(f'STD of C2-C3 CSA for each {across}')
    elif metric == "swdice":
        plt.ylabel('STD Dice')
        plt.title(f'STD of average slicewise Dice scores for each {across}')

    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')

    # Get y-axis limits
    ymin, ymax = plt.gca().get_ylim()

    # Draw vertical line between 1st and 2nd violin
    plt.axvline(x=0.5, color='k', linestyle='--')

    if across == "Method":
        # Compute the mean +- std across resolutions for each method and place it above the corresponding violin
        for method in df['Method'].unique():
            mean = df[df['Method'] == method]['std'].mean()
            std = df[df['Method'] == method]['std'].std()
            plt.text(hue_order.index(method), ymax-1, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')
    elif across == "Resolution":
        # Compute the mean +- std across resolutions for each method and place it above the corresponding violin
        for res in df['Resolution'].unique():
            mean = df[df['Resolution'] == res]['std'].mean()
            std = df[df['Resolution'] == res]['std'].std()
            plt.text(hue_order.index(res), ymax-0.5, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')
    else:
        raise ValueError(f'Unknown analysis type: {across}. Choices: [Method, Resolution, Threshold].')

    # Save the figure in 300 DPI as a PNG file
    if across == "Resolution":
        save_figure(file_path, f"std_csa_{across.lower()}_{model}.png")
    else:
        save_figure(file_path, f"std_csa_{across.lower()}.png")


def generate_figure_abs_csa_error(folder_path, data, hue_order=None):
    """
    Generate violinplot showing absolute CSA error across participants for each method
    """

    df = pd.DataFrame()
    for method in hue_order[1:]:
        df_error_contrast = pd.DataFrame()
        for contrast in CONTRAST_ORDER:
            df1 = data[(data['Method'] == "softseg_bin") & (data['Contrast'] == contrast)]
            df2 = data[(data['Method'] == method) & (data['Contrast'] == contrast)]

            # compute the slice-wise absolute error between the two dataframes
            df_temp = pd.merge(df1, df2, on=['Participant', 'Contrast', 'Slice (I->S)'], suffixes=('_gt', '_model'))

            # subtract the mean CSA of the ground truth from the mean CSA of the model based on Slice (I->S)
            df_temp['abs_error_per_slice'] = abs(df_temp['MEAN(area)_gt'] - df_temp['MEAN(area)_model'])

            # compute mean of the absolute error per slice for each participant
            df_error_contrast[contrast] = df_temp.groupby('Participant')['abs_error_per_slice'].mean()
        
        df_error_contrast['abs_error_mean'] = df_error_contrast.mean(axis=1)
        df_error_contrast['abs_error_std'] = df_error_contrast.std(axis=1)
        df_error_contrast['Method'] = method
        # df_error_contrast['Participant'] = df_temp['Participant']

        df = pd.concat([df, df_error_contrast])
        
    # remove the contrasts from the dataframe
    df = df.drop(columns=CONTRAST_ORDER)
    # # compute the mean and std across contrasts for each method
    # df_agg = df.groupby('Method')[['mean_error', 'std_error']].mean().reset_index()
    # print(df_agg)

    # remove softseg_bin from the list of methods for plotting
    hue_new = hue_order[1:]
    df = df[df['Method'].isin(hue_new)]

    plt.figure(figsize=(12, 6))
    # skip the first method (i.e., softseg_bin)
    sns.violinplot(x='Method', y='abs_error_mean', data=df, order=hue_new)
    # overlay swarm plot on the violin plot to show individual data points
    sns.swarmplot(x='Method', y='abs_error_mean', data=df, color='k', order=hue_new, size=3)

    plt.xlabel(None)    # plt.xlabel(across, fontsize=FONTSIZE)
    plt.ylabel('Absolute CSA error [mm^2]', fontweight='bold' ,fontsize=FONTSIZE)
    plt.title(f'Per-Slice Absolute CSA error between softseg_bin and other methods', 
              fontweight='bold' ,fontsize=FONTSIZE)
    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')

    # rename the x-axis ticks as per XTICKS
    plt.xticks(range(len(hue_new)), XTICKS[1:], fontsize=FONTSIZE)

    # Get y-axis limits
    ymin, ymax = plt.gca().get_ylim()

    # Compute the mean +- std across resolutions for each method and place it above the corresponding violin
    for method in df['Method'].unique():
        mean = df[df['Method'] == method]['abs_error_mean'].mean()
        std = df[df['Method'] == method]['abs_error_std'].mean()
        plt.text(hue_new.index(method), ymax-1, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')
    
    # Save the figure in 300 DPI as a PNG file
    save_figure(folder_path, "abs_csa_error_perslice.png")


def generate_figure_abs_csa_error_per_contrast(file_path, data, method=None, threshold=None):
    """
    Generate violinplot showing absolute CSA error for each contrast for a given method/threshold
    """

    # remove "softseg_soft" from the list of methods, if it exists
    if 'softseg_soft' in data['Method'].unique():
        data = data[data['Method'] != 'softseg_soft']

    if method is not None and threshold is None:
        # create a dataframe with only "softseg_bin" and get the mean CSA for each contrast
        df1 = data[data['Method'] == "softseg_bin"].groupby(['Contrast', 'Participant'])['MEAN(area)'].agg(['mean']).reset_index()

        # create a dataframe with only the given method and get the mean CSA for each contrast
        df2 = data[data['Method'] == method].groupby(['Contrast', 'Participant'])['MEAN(area)'].agg(['mean']).reset_index()

        # define title for the plot
        title = f'Method: {method}; Absolute CSA error for each contrast'

        # define save path for the plot
        save_fname = f"abs_error_per_contrast_{method}.png"
        
    elif method is None and threshold is not None:
        # create a dataframe for the given threshold
        df_thr = data[data['Threshold'] == threshold]

        # create a dataframe with only "softseg" and get the mean CSA for each contrast
        df1 = df_thr[df_thr['Method'] == "softseg"].groupby(['Contrast', 'Participant'])['MEAN(area)'].agg(['mean']).reset_index()

        # create a dataframe with only "monai" and get the mean CSA for each contrast
        df2 = df_thr[df_thr['Method'] == "monai"].groupby(['Contrast', 'Participant'])['MEAN(area)'].agg(['mean']).reset_index()

        # define title for the plot
        title = f'Threshold: 0.{threshold}; Absolute CSA error between softseg and monai preds for each contrast'

        # define save path for the plot
        save_fname = f"abs_error_per_contrast_{threshold}.png"

    df1 = df1.pivot(index='Participant', columns='Contrast', values='mean').reset_index()
    df2 = df2.pivot(index='Participant', columns='Contrast', values='mean').reset_index()

    # compute the absolute error between the two dataframes for each contrast
    df = pd.DataFrame()
    df['Participant'] = df1['Participant']
    for contrast in CONTRAST_ORDER:
        df[contrast] = abs(df1[contrast] - df2[contrast])
    
    # reshape the dataframe to have a single column for the contrast and a single column for the absolute error
    df = df.melt(id_vars=['Participant'], value_vars=CONTRAST_ORDER, var_name='Contrast', value_name='abs_error')

    # plot the abs error for each contrast in a violinplot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Contrast', y='abs_error', data=df, order=CONTRAST_ORDER)
    # overlay swarm plot on the violin plot to show individual data points
    sns.swarmplot(x='Contrast', y='abs_error', data=df, color='k', order=CONTRAST_ORDER, size=3)
    plt.xlabel('Contrast')
    plt.ylabel('Absolute CSA error [mm^2]')
    plt.title(title)
    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')
    # set the y-axis limits
    plt.ylim(-2, 10)

    # Get y-axis limits
    ymin, ymax = plt.gca().get_ylim()

    # Compute the mean +- std across resolutions for each method and place it above the corresponding violin
    for contrast in CONTRAST_ORDER:
        mean = df[df['Contrast'] == contrast]['abs_error'].mean()
        std = df[df['Contrast'] == contrast]['abs_error'].std()
        plt.text(CONTRAST_ORDER.index(contrast), ymax-1, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')

    # Save the figure in 300 DPI as a PNG file
    save_figure(file_path, save_fname)



def compute_cov(data, file_path):
    """
    Compute COV for CSA for each method across resolutions
    :param data: Pandas DataFrame with the data
    :param file_path: Path to the CSV file (will be used to save the figure)
    """
    # Compute COV for CSA ('MEAN(area)' column) for each method across resolutions
    df = data.groupby(['Method', 'Resolution'])['MEAN(area)'].agg(['mean', 'std']).reset_index()
    for method in df['Method'].unique():
        df.loc[df['Method'] == method, 'COV'] = df.loc[df['Method'] == method, 'std'] / df.loc[
            df['Method'] == method, 'mean'] * 100
    df = df[['Method', 'Resolution', 'COV']].pivot(index='Resolution', columns='Method', values='COV')
    # Compute mean +- std across resolutions for each method
    df.loc['mean COV'] = df.mean()
    df.loc['std COV'] = df.std()
    # Keep only two decimals and save as csv
    df = df.round(2)
    df.to_csv(file_path.replace('.csv', '_COV.csv'))
    print(f'COV saved to {file_path.replace(".csv", "_COV.csv")}')
    # Print
    print(df)


def main(args, analysis_type="methods"):

    csvs_list = [f for f in os.listdir(args.i_folder) if f.endswith('_perslice.csv')]
    # initialize a new dataframe
    df_final = pd.DataFrame()
    for csv in csvs_list:
        # load the CSV file containing per slice CSAs for each method (separately)
        data = pd.read_csv(os.path.join(args.i_folder, csv))

        # apply the function to extract participant ID
        data['Participant'] = data['Filename'].apply(fetch_participant_id)

        data['Contrast'], data['Method'] = zip(
            *data['Filename'].apply(extract_contrast_and_details, across="Method"))

        # create a new dataframe with columns only Participant, Contrast, Method
        data_new = data[['Participant', 'Contrast', 'Method', 'Slice (I->S)', 'MEAN(area)']]
        data_new = data_new.drop_duplicates().reset_index(drop=True)
        # replace nans with 0
        data_new = data_new.fillna(0.0)

        # concatenate the dataframes
        df_final = pd.concat([df_final, data_new])
                
    # Apply the function to extract method and the corresponding analysis details
    if analysis_type == "methods":
        # Generate violinplot showing absolute CSA error across participants for each method
        generate_figure_abs_csa_error(args.i_folder, df_final, hue_order=HUE_ORDER)

        # # Generate violinplot showing absolute CSA error for each contrast for a given method
        # for method in HUE_ORDER[1:]:
        #     generate_figure_abs_csa_error_per_contrast(file_path, data, method=method, threshold=None)

    elif analysis_type == "resolutions":
        
        # Generate violinplot showing STD across participants for each resolution
        # generate_figure_std(data, file_path, across="Resolution", hue_order=HUE_ORDER_RES)
        generate_figure_std(df_final, args.i_folder, across="Method", hue_order=HUE_ORDER[1:])
    
    else:
        raise ValueError(f'Unknown analysis type: {analysis_type}. Choices: [methods, resolutions, thresholds].')

    # # Compute COV for CSA for each method across resolutions
    # compute_cov(data, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate violin plot from CSV data.')
    parser.add_argument('-i-folder', type=str, 
                        help='Path to the folder containing CSV files with per slice CSA')
    parser.add_argument('-a', type=str, default="methods", 
                        help='Options to analyse CSA across. Choices: [methods, resolutions]')
    args = parser.parse_args()
    main(args, analysis_type=args.a)