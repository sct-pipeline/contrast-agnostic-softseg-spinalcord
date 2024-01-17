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
# HUE_ORDER = ["MTon", "MToff", "T1w", "T2star", "T2w"]
# HUE_ORDER = ["2e-01", "1e-01", "1e-02", "1e-03", "1e-04"]
HUE_ORDER = ["GT", "2e-01", "1e-01", "1e-02", "1e-04"]

# Function to extract method and resolution from the filename
def extract_contrast_and_threshold(filename, file_type='pred'):
    """
    Extract the segmentation contrast and the threshold from from the filename.
    The method (e.g., propseg, deepseg_2d, nnunet_3d_fullres, monai) and resolution (e.g., 1mm)
    are embedded in the filename.
    """
    # pattern = r'.*iso-(\d+mm).*_(propseg|deepseg_2d|nnunet_3d_fullres|monai_bin).*'
    if file_type == 'pred':
        pattern = r'.*_(MTon|MToff|T1w|T2star|T2w).*_(\d+e-\d+).*'
    elif file_type == 'gt':
        pattern = r'.*_(MTon|MToff|T1w|T2star|T2w).*'
    else:
        raise ValueError(f'Unknown file type: {file_type}')
    match = re.search(pattern, filename)
    if match:
        if file_type == 'pred':
            return match.group(1), match.group(2)
        elif file_type == 'gt':
            # NOTE: this is a hack to return 'GT' as the threshold so that the GT dataframe
            # can be merged with the predictions dataframe and a single violinplot can be generated
            return match.group(1), 'GT'
    else:
        return 'Unknown', 'Unknown'


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


def generate_figure(data, file_path):
    """
    Generate violinplot across resolutions and methods
    :param data: Pandas DataFrame with the data
    :param contrast: Contrast (e.g., T1w, T2w, T2star)
    :param file_path: Path to the CSV file (will be used to save the figure)
    """

    # Correct labels for the x-axis based on the actual data
    # resolution_labels = ['1mm', '05mm', '15mm', '4mm', '2mm']
    # threshold_labels = ['2e-01', '1e-01', '1e-02', '1e-03', '1e-04']
    contrast_labels = ['MTon', 'MToff', 'T1w', 'T2star', 'T2w']
    
    # Creating the violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Contrast', y='MEAN(area)', hue='Threshold', data=data, order=contrast_labels,
                   hue_order=HUE_ORDER)

    # plt.xticks(rotation=45)
    plt.xlabel('Contrast')
    plt.ylabel('CSA [mm^2]')
    plt.title(f'C2-C3 CSA across Contrasts and Thresholds')
    plt.legend(title='Threshold', loc='lower left')
    plt.tight_layout()

    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')

    # # Update x-axis labels
    # plt.gca().set_xticklabels(['1x1x1mm', '0.5x0.5x4mm', '1.5x1.5x1.5mm', '4x0.5x0.5mm', '2x2x2mm'])

    # Save the figure in 300 DPI as a PNG file
    plt.savefig(file_path.replace('.csv', '.png'), dpi=300)
    print(f'Figure saved to {file_path.replace(".csv", ".png")}')

    # Display the plot
    plt.show()


def generate_figure_std(data, contrast, file_path):
    """
    Generate violinplot showing STD across participants for each method
    """

    # Compute mean and std across resolutions for each method
    df = data.groupby(['Method', 'Participant'])['MEAN(area)'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Method', y='std', data=df, order=HUE_ORDER)
    # plt.xticks(rotation=45)
    plt.xlabel('Method')
    plt.ylabel('STD [mm^2]')
    plt.title(f'{contrast}: STD of C2-C3 CSA across resolutions for each method')
    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')

    # Get y-axis limits
    ymin, ymax = plt.gca().get_ylim()

    # Compute the mean +- std across resolutions for each method and place it above the corresponding violin
    for method in df['Method'].unique():
        mean = df[df['Method'] == method]['std'].mean()
        std = df[df['Method'] == method]['std'].std()
        plt.text(HUE_ORDER.index(method), ymax-3, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')

    # Save the figure in 300 DPI as a PNG file
    plt.tight_layout()
    plt.savefig(file_path.replace('.csv', '_STD.png'), dpi=300)
    print(f'Figure saved to {file_path.replace(".csv", "_STD.png")}')


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


def main(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # replace occurences
    data['Filename'] = data['Filename'].str.replace('flip-2_mt-off_MTS', 'MToff')
    data['Filename'] = data['Filename'].str.replace('flip-1_mt-on_MTS', 'MTon')

    # Apply the function to extract method and resolution
    data['Contrast'], data['Threshold'] = zip(*data['Filename'].apply(extract_contrast_and_threshold, file_type='pred'))

    # remove rows with threshold 1e-03
    data = data[data['Threshold'] != '1e-03']

    # Apply the function to extract participant ID
    data['Participant'] = data['Filename'].apply(fetch_participant_id)

    # ---------------------------------------------------------------------------------------------
    # load the GT CSA CSV files
    contrasts = ['MTon', 'MToff', 'T1w', 'T2star', 'T2w']
    data_gt = pd.DataFrame()
    for contrast in contrasts:
        # load the GT CSA data and merge it into one main dataframe
        gt_csa_file_path = os.path.join(args.i_gt, f"csa_soft_GT_{contrast}.csv")
        gt_csa_data = pd.read_csv(gt_csa_file_path)
        data_gt = pd.concat([data_gt, gt_csa_data], axis=0, ignore_index=True)

    # apply the function to extract participant ID
    data_gt['Participant'] = data_gt['Filename'].apply(fetch_participant_id)

    # keep only those participants in data_gt that are also in data (i.e. the test set)
    data_gt = data_gt[data_gt['Participant'].isin(data['Participant'])]

    # save the GT CSA data
    data_gt.to_csv(os.path.join(args.i_gt, f"csa_soft_GT_all_contrasts_test.csv"), index=False)

    # replace occurences
    data_gt['Filename'] = data_gt['Filename'].str.replace('flip-2_mt-off_MTS', 'MToff')
    data_gt['Filename'] = data_gt['Filename'].str.replace('flip-1_mt-on_MTS', 'MTon')

    # Apply the function to extract contrast from GT CSA data
    data_gt['Contrast'], data_gt['Threshold'] = zip(*data_gt['Filename'].apply(extract_contrast_and_threshold, file_type='gt'))

    # merge the GT CSA data into the main dataframe
    data_preds_and_gt = pd.concat([data, data_gt], axis=0, ignore_index=True)

    # Generate violinplot across resolutions and methods
    generate_figure(data_preds_and_gt, file_path)

    # # Generate violinplot showing STD across participants for each method
    # generate_figure_std(data, contrast, file_path)

    # # Compute COV for CSA for each method across resolutions
    # compute_cov(data, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate violin plot from CSV data.')
    parser.add_argument('-i', type=str, help='Path to the CSV file')
    parser.add_argument('-i-gt', type=str, help='Path to the folder with the GT CSA CSV files')
    args = parser.parse_args()
    main(args.i)
