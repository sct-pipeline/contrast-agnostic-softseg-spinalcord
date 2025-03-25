"""
Generate violin plots from CSV data across different model releases/versions.

Usage:
    python analyse_csa_across_releases.py -i /path/to/data.csv -a [methods, resolutions, thresholds]
"""

import os
import argparse
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# Setting the hue order as specified
FONTSIZE = 12
CONTRAST_ORDER = ["DWI", "MTon", "MToff", "T1w", "T2star", "T2w"]


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
def extract_contrast_and_details(filename, model_versions):
    """
    Extract the segmentation method (e.g., deepseg (2d), v2.0 (monai), v3.0 (nnunet)) from the filename.
    """
    # pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg_bin|deepseg|).*'
    pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg_bin|).*'

    # Extract existing values in the second capturing group
    match = re.search(r'_\(softseg_bin\|?(.*?)\)\._', pattern)  # Extract contents inside (softseg_bin|...)
    existing_versions = set(match.group(1).split('|')) if match and match.group(1) else set()

    # Merge existing versions with new model versions
    updated_versions = existing_versions.union(model_versions)

    # Rebuild the pattern, updating only the second capturing group
    updated_pattern = re.sub(
        r'\(softseg_bin\|?.*?\)',  # Match the existing group
        f"(softseg_bin|{'|'.join(sorted(updated_versions))})",  # Replace with updated group
        pattern
    )    

    match = re.search(updated_pattern, filename)
    if match:
        return match.group(1), match.group(2)
    else:
        return 'Unknown', 'Unknown'    


def generate_figure_csa(file_path, data, method=None):
    """
    Generate violinplot showing absolute CSA error for each contrast for a given method/threshold
    """

    if method is not None:

        # create a dataframe with only the given method and get the mean CSA for each contrast
        df = data[data['Method'] == method].groupby(['Contrast', 'Participant'])['MEAN(area)'].agg(['mean']).reset_index()

        # define title for the plot
        title = f'Method: {method}; CSA across MRI contrasts'

        # define save path for the plot
        save_fname = f"csa__model_{method}.png"
        
    # plot the abs error for each contrast in a violinplot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Contrast', y='mean', data=df, order=CONTRAST_ORDER, hue='Contrast', legend=False)
    # overlay swarm plot on the violin plot to show individual data points
    sns.swarmplot(x='Contrast', y='mean', data=df, color='k', order=CONTRAST_ORDER, size=3)
    plt.xlabel(None)
    plt.ylabel('CSA [mm^2]', fontweight='bold' ,fontsize=FONTSIZE)
    plt.title(title)
    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')
    # set the y-axis limits
    plt.ylim(40, 110)
    plt.yticks(range(40, 110, 5))

    plt.xticks(range(len(CONTRAST_ORDER)), CONTRAST_ORDER, fontsize=FONTSIZE)

    # Get y-axis limits
    ymin, ymax = plt.gca().get_ylim()

    # Compute the mean +- std across resolutions for each method and place it above the corresponding violin
    for contrast in CONTRAST_ORDER:
        mean = df[df['Contrast'] == contrast]['mean'].mean()
        std = df[df['Contrast'] == contrast]['mean'].std()
        plt.text(CONTRAST_ORDER.index(contrast), ymax-5, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k', fontsize=FONTSIZE)

    # Save the figure in 300 DPI as a PNG file
    save_path = os.path.join(file_path, save_fname)
    plt.savefig(save_path, dpi=300)
    print(f'Figure saved to {save_path}')



def generate_figure_std_csa(data, file_path, across="Method", hue_order=None):
    """
    Generate violinplot showing STD across participants for each method
    """
    
    # Compute mean and std across contrasts for each method
    df = data.groupby([across, 'Participant'])['MEAN(area)'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(12, 6))
    sns.violinplot(x=across, y='std', data=df, order=hue_order, hue=across, legend=False)
    # overlay swarm plot on the violin plot to show individual data points
    sns.swarmplot(x=across, y='std', data=df, color='k', order=hue_order, size=3)

    # Draw vertical line between 1st and 2nd violin
    plt.axvline(x=0.5, color='k', linestyle='--')

    plt.xlabel(None)    # plt.xlabel(across)
    plt.ylabel('STD [mm^2]', fontweight='bold' ,fontsize=FONTSIZE)
    plt.title(f'STD of C2-C3 CSA for each {across}', fontweight='bold' ,fontsize=FONTSIZE)

    XTICKS = [hue_order[0].replace('softseg_bin', 'GT')] + hue_order[1:]
    XTICKS = [XTICKS[0]] + [f"model_{version}" for version in XTICKS[1:]]
    plt.xticks(range(len(hue_order)), XTICKS, fontsize=FONTSIZE)

    # set upper y-axis limits
    plt.ylim(-0.5, 8.5)

    # Get y-axis limits
    ymin, ymax = plt.gca().get_ylim()

    # Compute the mean +- std across resolutions for each method and place it above the corresponding violin
    for method in df['Method'].unique():
        mean = df[df['Method'] == method]['std'].mean()
        std = df[df['Method'] == method]['std'].std()
        plt.text(hue_order.index(method), ymax-1, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')

    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')

    # Save the figure in 300 DPI as a PNG file
    save_path = os.path.join(file_path, f"std_c2c3_csa_across_versions.png")
    plt.savefig(save_path, dpi=300)
    print(f'Figure saved to {save_path}')


def main(args):

    csvs_list = glob.glob(os.path.join(args.i, "*.csv"))
    # sort the list of CSV files
    csvs_list.sort()

    # only take the most recent 5 models for comparison (irrespective of how many are released)
    csvs_list = csvs_list[-5:]
    
    # of the format: csa_c2c3__model_v2.0.csv, csa_c2c3__model_v3.0.csv
    models_to_compare = [os.path.basename(f).split('__')[1].strip('.csv') for f in csvs_list]
    model_versions = [model.split('_')[1] for model in models_to_compare]

    # define order of the violin plots
    hue_order = ['softseg_bin'] + model_versions
    
    # merge the CSV files for each model release
    if len(csvs_list) > 1:
        data_avg_csa = pd.concat([pd.read_csv(f) for f in csvs_list], ignore_index=True)
    else:
        data_avg_csa = pd.read_csv(csvs_list[0])

    # Apply the function to extract participant ID
    data_avg_csa['Participant'] = data_avg_csa['Filename'].apply(fetch_participant_id)

    # Apply the function to extract method and the corresponding analysis details
    data_avg_csa['Contrast'], data_avg_csa['Method'] = zip(
        *data_avg_csa['Filename'].apply(extract_contrast_and_details, model_versions=model_versions))
        
    # Generate violinplot showing STD across participants for each method
    generate_figure_std_csa(data_avg_csa, file_path=args.i, across="Method", hue_order=hue_order)

    # Generate violinplot showing absolute CSA error for each contrast for a given method/threshold
    for method in model_versions:
        generate_figure_csa(file_path=args.i, data=data_avg_csa, method=method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate violin plot from CSV data.')
    parser.add_argument('-i', type=str, required=True,
                        help='Path to the folder containing CSV files C2-C3 CSA for all releases.'
                             'Output will be saved in the same folder')

    args = parser.parse_args()
    main(args)