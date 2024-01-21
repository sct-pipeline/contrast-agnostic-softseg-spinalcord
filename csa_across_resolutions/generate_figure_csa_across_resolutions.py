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
HUE_ORDER = ["propseg", "deepseg_2d", "nnunet_3d_fullres", "monai"]

# Function to extract method and resolution from the filename
def extract_method_resolution(filename):
    """
    Extract the segmentation method and resolution from the filename.
    The method (e.g., propseg, deepseg_2d, nnunet_3d_fullres, monai) and resolution (e.g., 1mm)
    are embedded in the filename.
    """
    pattern = r'.*iso-(\d+mm).*_(propseg|deepseg_2d|nnunet_3d_fullres|monai).*'
    match = re.search(pattern, filename)
    if match:
        return match.group(2), match.group(1)
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
        plt.text(HUE_ORDER.index(method), ymax-1, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')

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

    # Apply the function to extract method and resolution
    data['Method'], data['Resolution'] = zip(*data['Filename'].apply(extract_method_resolution))

    # Apply the function to extract participant ID
    data['Participant'] = data['Filename'].apply(fetch_participant_id)

    # Fetch contrast (e.g. T1w, T2w, T2star) from the first filename using regex
    contrast = re.search(r'.*_(T1w|T2w|T2star).*', data['Filename'][0]).group(1)

    # Generate violinplot across resolutions and methods
    generate_figure(data, contrast, file_path)

    # Generate violinplot showing STD across participants for each method
    generate_figure_std(data, contrast, file_path)

    # Compute COV for CSA for each method across resolutions
    compute_cov(data, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate violin plot from CSV data.')
    parser.add_argument('-i', type=str, help='Path to the CSV file')
    args = parser.parse_args()
    main(args.i)
