"""
Generate violin plot from CSV data across resolutions and methods.

Usage:
    python generate_figure_csa_across_resolutions.py -i /path/to/data.csv
"""

import os
import argparse
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Setting the hue order as specified
# HUE_ORDER = ["softseg_soft", "softseg_bin", "nnunet", "monai_soft", "monai_bin"]
HUE_ORDER = ["softseg_bin", "deepseg_2d", "nnunet", "monai", "swinunetr", "mednext"]
HUE_ORDER_THR = ["GT", "15", "1", "05", "01", "005"]
CONTRAST_ORDER = ["DWI", "MTon", "MToff", "T1w", "T2star", "T2w"]


def save_figure(file_path, save_fname):
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(file_path), save_fname)
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
def extract_contrast_and_details(filename, analysis_type):
    """
    Extract the segmentation method and resolution from the filename.
    The method (e.g., propseg, deepseg_2d, nnunet_3d_fullres, monai) and resolution (e.g., 1mm)
    are embedded in the filename.
    """
    # pattern = r'.*iso-(\d+mm).*_(propseg|deepseg_2d|nnunet_3d_fullres|monai).*'
    # pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg_soft|softseg_bin|nnunet|monai_soft|monai_bin).*'
    if analysis_type == "methods":
        pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg_bin|deepseg_2d|nnunet|monai|swinunetr|mednext).*'
        match = re.search(pattern, filename)
        if match:
            return match.group(1), match.group(2)
        else:
            return 'Unknown', 'Unknown'

    elif analysis_type == "thresholds":
        pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg|monai)_thr_(\d+).*'
        match = re.search(pattern, filename)
        if match:
            return match.group(1), match.group(2), match.group(3)
        else:
            return 'Unknown', 'Unknown', 'Unknown'

    elif analysis_type == "resolutions":
        pattern = r'.*iso-(\d+mm).*_(softseg_bin|deepseg_2d|nnunet|monai|swinunetr|mednext).*'
        # TODO

    else:
        raise ValueError(f'Unknown analysis type: {analysis_type}. Choices: [methods, resolutions, thresholds].')
    


def generate_figure(data, contrast, file_path):
    """
    Generate violinplot across resolutions and methods
    :param data: Pandas DataFrame with the data
    :param contrast: Contrast (e.g., T1w, T2w, T2star)
    :param file_path: Path to the CSV file (will be used to save the figure)
    """

    # Correct labels for the x-axis based on the actual data
    # resolution_labels = ['1mm', '125mm', '15mm', '175mm', '2mm']
    resolution_labels = ['1mm', '05mm', '15mm', '3mm', '2mm']

    # Creating the violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Resolution', y='MEAN(area)', hue='Method', data=data, order=resolution_labels,
                   hue_order=HUE_ORDER)
    plt.xticks(rotation=45)
    plt.xlabel('Resolution')
    plt.ylabel('CSA [mm^2]')
    plt.title(f'{contrast}: C2-C3 CSA across Resolutions and Methods')
    plt.legend(title='Method', loc='lower left')
    plt.tight_layout()

    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')

    # Update x-axis labels
    # plt.gca().set_xticklabels(['1mm', '1.25mm', '1.5mm', '1.75mm', '2mm'])
    plt.gca().set_xticklabels(['1x1x1mm', '0.5x0.5x0.5mm', '1.5mm', '3x0.5x0.5mm', '2mm'])

    # Save the figure in 300 DPI as a PNG file
    plt.savefig(file_path.replace('.csv', '.png'), dpi=300)
    print(f'Figure saved to {file_path.replace(".csv", ".png")}')

    # Display the plot
    plt.show()


def generate_figure_std(data, file_path):
    """
    Generate violinplot showing STD across participants for each method
    """

    # Compute mean and std across contrasts for each method
    df = data.groupby(['Method', 'Participant'])['MEAN(area)'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Method', y='std', data=df, order=HUE_ORDER)
    # overlay swarm plot on the violin plot to show individual data points
    sns.swarmplot(x='Method', y='std', data=df, color='k', order=HUE_ORDER, size=3)
    # plt.xticks(rotation=45)
    plt.xlabel('Method')
    plt.ylabel('STD [mm^2]')
    plt.title(f'STD of C2-C4 CSA for each method')
    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')

    # Get y-axis limits
    ymin, ymax = plt.gca().get_ylim()

    # Draw vertical line between 1st and 2nd violin
    plt.axvline(x=0.5, color='k', linestyle='--')

    # Compute the mean +- std across resolutions for each method and place it above the corresponding violin
    for method in df['Method'].unique():
        mean = df[df['Method'] == method]['std'].mean()
        std = df[df['Method'] == method]['std'].std()
        plt.text(HUE_ORDER.index(method), ymax-1, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')

    # Save the figure in 300 DPI as a PNG file
    save_figure(file_path, "std_csa.png")


def generate_figure_abs_csa_error(file_path, data, hue_order=None):
    """
    Generate violinplot showing absolute CSA error across participants for each method
    """

    # Remove "softseg_soft" from the list of methods, if it exists
    if 'softseg_soft' in data['Method'].unique():
        data = data[data['Method'] != 'softseg_soft']

    # Compute mean and std across contrasts for each method
    df = data.groupby(['Method', 'Participant'])['MEAN(area)'].agg(['mean', 'std']).reset_index()
    
    # Compute the abs error between "sofseg_bin" and all other methods
    df['abs_error'] = df.apply(lambda row: abs(row['mean'] - df[(df['Method'] == 'softseg_bin') & (df['Participant'] == row['Participant'])]['mean'].values[0]), axis=1)

    # Remove "softseg_bin" from the list of methods and shift rows by one to match the violinplot
    df = df[df['Method'] != 'softseg_bin']

    plt.figure(figsize=(12, 6))
    # skip the first method (i.e., softseg_bin)
    sns.violinplot(x='Method', y='abs_error', data=df, order=hue_order)
    # overlay swarm plot on the violin plot to show individual data points
    sns.swarmplot(x='Method', y='abs_error', data=df, color='k', order=hue_order, size=3)

    # plt.xticks(rotation=45)
    plt.xlabel('Method')
    plt.ylabel('Absolute CSA error [mm^2]')
    plt.title(f'Absolute CSA error between softseg_bin and other methods')
    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')

    # Get y-axis limits
    ymin, ymax = plt.gca().get_ylim()

    # Draw vertical line between 1st and 2nd violin
    plt.axvline(x=0.5, color='k', linestyle='--')

    # Compute the mean +- std across resolutions for each method and place it above the corresponding violin
    for method in df['Method'].unique():
        mean = df[df['Method'] == method]['abs_error'].mean()
        std = df[df['Method'] == method]['abs_error'].std()
        plt.text(hue_order.index(method), ymax-0.25, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')
    
    # Save the figure in 300 DPI as a PNG file
    save_figure(file_path, "abs_csa_error.png")


def generate_figure_abs_csa_error_threshold(file_path, data, hue_order=None):
    """
    Generate violinplot showing absolute CSA error across participants for each threshold value
    """

    # Compute mean and std across thresholds for each method
    df = data.groupby(['Method', 'Threshold', 'Participant'])['MEAN(area)'].agg(['mean', 'std']).reset_index()

    # compute abs_error between softseg and monai for each threshold across all contrasts
    df['abs_error'] = df.apply(lambda row: abs(row['mean'] - df[(df['Method'] == 'softseg') & (df['Threshold'] == row['Threshold']) & (df['Participant'] == row['Participant'])]['mean'].values[0]), axis=1)

    # remove "softseg" from the list of methods
    df = df[df['Method'] != 'softseg']

    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Threshold', y='abs_error', data=df, order=hue_order)
    # overlay swarm plot on the violin plot to show individual data points
    sns.swarmplot(x='Threshold', y='abs_error', data=df, color='k', order=hue_order, size=3)
    plt.xlabel('Threshold')
    plt.ylabel('Absolute CSA error [mm^2]')
    plt.title(f'Absolute CSA error between softseg and monai for each threshold')
    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')

    # Get y-axis limits
    ymin, ymax = plt.gca().get_ylim()

    # Compute the mean +- std across thresholds for each method and place it above the corresponding violin
    for threshold in HUE_ORDER_THR:
        mean = df[df['Threshold'] == threshold]['abs_error'].mean()
        std = df[df['Threshold'] == threshold]['abs_error'].std()
        plt.text(HUE_ORDER_THR.index(threshold), ymax-1, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')

    # Save the figure in 300 DPI as a PNG file
    save_figure(file_path, "abs_csa_error_threshold.png")


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


def main(file_path, analysis_type="methods"):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Apply the function to extract participant ID
    data['Participant'] = data['Filename'].apply(fetch_participant_id)

    # Apply the function to extract method and the corresponding analysis details
    if analysis_type == "methods":
        data['Contrast'], data['Method'] = zip(
            *data['Filename'].apply(extract_contrast_and_details, analysis_type=analysis_type))

        # Generate violinplot showing STD across participants for each method
        generate_figure_std(data, file_path)

        # Generate violinplot showing absolute CSA error across participants for each method
        generate_figure_abs_csa_error(file_path, data, hue_order=HUE_ORDER)

        # Generate violinplot showing absolute CSA error for each contrast for a given method
        for method in HUE_ORDER[1:]:
            generate_figure_abs_csa_error_per_contrast(file_path, data, method=method, threshold=None)

    elif analysis_type == "thresholds":
        data['Contrast'], data['Method'], data['Threshold'] = zip(
            *data['Filename'].apply(extract_contrast_and_details, analysis_type=analysis_type))

        # Generate violinplot showing absolute CSA error across participants for each threshold value
        generate_figure_abs_csa_error_threshold(file_path, data, hue_order=HUE_ORDER_THR)

        # Generate violinplot showing absolute CSA error for each contrast for a given threshold
        for threshold in HUE_ORDER_THR[1:]:
            generate_figure_abs_csa_error_per_contrast(file_path, data, method=None, threshold=threshold)

    elif analysis_type == "resolutions":
        # TODO
        pass
    
    else:
        raise ValueError(f'Unknown analysis type: {analysis_type}. Choices: [methods, resolutions, thresholds].')

    # # Compute COV for CSA for each method across resolutions
    # compute_cov(data, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate violin plot from CSV data.')
    parser.add_argument('-i', type=str, help='Path to the CSV file')
    parser.add_argument('-a', type=str, default="methods", 
                        help='Options to analyse CSA across. Choices: [methods, resolutions, thresholds]')
    args = parser.parse_args()
    main(args.i, analysis_type=args.a)