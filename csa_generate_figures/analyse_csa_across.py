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
# HUE_ORDER = ["softseg_bin", "deepseg", "plain_320", "plain_384", "resencM"]
HUE_ORDER = ["softseg_bin", "nnunet-AllRandInit2D", "nnunet-AllRandInit3D", "vPtrV21-allWithPraxWithSCT"]
# HUE_ORDER = ["softseg_bin", "v20", "v21", "v23", "v24", "v25", "deepseg", "vPtrV21noPrax"]
HUE_ORDER_THR = ["GT", "15", "1", "05", "01", "005"]
HUE_ORDER_RES = ["1mm", "05mm", "15mm", "3mm", "2mm"]
CONTRAST_ORDER = ["DWI", "MTon", "MToff", "T1w", "T2star", "T2w"]

FONTSIZE = 12
XTICKS = ["GT", "C-A\nnnunet-AllRandInit2D", "C-A\nnnunet-AllRandInit3D", "C-A-vPtr2.1\nM5prime"]
# XTICKS = ["GT", "contrast-agnostic\nv2.0", "contrast-agnostic\nv2.1", "contrast-agnostic\nv2.3",
#             "contrast-agnostic\nv2.4", "contrast-agnostic\nv2.5", "deepseg_sc\n2D", "contrast-agnostic\nvPtrV21noPrax",]


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
def extract_contrast_and_details(filename, across="Method"):
    """
    Extract the segmentation method and resolution from the filename.
    The method (e.g., propseg, deepseg_2d, nnunet_3d_fullres, monai) and resolution (e.g., 1mm)
    are embedded in the filename.
    """
    # pattern = r'.*iso-(\d+mm).*_(propseg|deepseg_2d|nnunet_3d_fullres|monai).*'
    if across == "Method":
        # pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg_bin|deepseg_2d|nnunet|monai|mednext|swinunetr|swinpretrained|ensemble).*'
        # pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg_bin|deepseg|plain_320|plain_384|resencM).*'
        pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg_bin|deepseg|v20|v21|v23|v24|v25|nnunet-AllRandInit2D|nnunet-AllRandInit3D|vPtrV21-allWithPraxWithSCT).*'
        match = re.search(pattern, filename)
        if match:
            return match.group(1), match.group(2)
        else:
            return 'Unknown', 'Unknown'

    elif across == "Threshold":
        pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w).*_(softseg|monai)_thr_(\d+).*'
        match = re.search(pattern, filename)
        if match:
            return match.group(1), match.group(2), match.group(3)
        else:
            return 'Unknown', 'Unknown', 'Unknown'

    elif across == "Resolution":
        pattern = r'.*_(DWI|MTon|MToff|T1w|T2star|T2w)_res_(\d+mm).*_(deepseg_2d|nnunet|monai|swinunetr|mednext).*'
        match = re.search(pattern, filename)
        if match:
            return match.group(1), match.group(3), match.group(2)
        else:
            return 'Unknown', 'Unknown', 'Unknown'

    else:
        raise ValueError(f'Unknown analysis type: {across}. Choices: [Method, Resolution, Threshold].')
    


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

    plt.figure(figsize=(12, 6))
    
    if metric == "csa":
        # Compute mean and std across contrasts for each method
        df = data.groupby([across, 'Participant'])['MEAN(area)'].agg(['mean', 'std']).reset_index()

        sns.violinplot(x=across, y='std', data=df, order=hue_order)
        # overlay swarm plot on the violin plot to show individual data points
        sns.swarmplot(x=across, y='std', data=df, color='k', order=hue_order, size=3)

        # Draw vertical line between 1st and 2nd violin
        plt.axvline(x=0.5, color='k', linestyle='--')

        plt.xlabel(None)    # plt.xlabel(across)
        plt.ylabel('STD [mm^2]', fontweight='bold' ,fontsize=FONTSIZE)
        plt.title(f'STD of C2-C3 CSA for each {across}', fontweight='bold' ,fontsize=FONTSIZE)

        plt.xticks(range(len(hue_order)), XTICKS, fontsize=FONTSIZE)        
        
        # Get y-axis limits
        ymin, ymax = plt.gca().get_ylim()

        if across == "Method":
            # Compute the mean +- std across resolutions for each method and place it above the corresponding violin
            for method in df['Method'].unique():
                mean = df[df['Method'] == method]['std'].mean()
                std = df[df['Method'] == method]['std'].std()
                plt.text(hue_order.index(method), ymax-1, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')


    elif metric == "swdice":
        # Compute mean and std across contrasts for each method
        df = data.groupby([across, 'Participant'])['average_slicewise_dice'].agg(['mean', 'std']).reset_index()
        hue_order = HUE_ORDER[1:]      # skip the first method (i.e., softseg_bin)
        df = df[df['Method'].isin(hue_order)]

        plot_values = 'std'
        sns.violinplot(x=across, y=plot_values, data=df, order=hue_order)
        # overlay swarm plot on the violin plot to show individual data points
        sns.swarmplot(x=across, y=plot_values, data=df, color='k', order=hue_order, size=3)
        
        plt.xlabel(None)    # plt.xlabel(across)
        plt.ylabel(f"{plot_values.upper()} Dice", fontweight='bold' ,fontsize=FONTSIZE)
        plt.title(f"{plot_values.upper()} of average slicewise Dice scores across contrasts for each {across}", 
                  fontweight='bold' ,fontsize=FONTSIZE)

        plt.xticks(range(len(hue_order)), XTICKS[1:], fontsize=FONTSIZE)        

        # Get y-axis limits
        ymin, ymax = plt.gca().get_ylim()

        if across == "Method":
            # Compute the mean +- std across resolutions for each method and place it above the corresponding violin
            for method in df['Method'].unique():
                mean = df[df['Method'] == method][plot_values].mean()
                std = df[df['Method'] == method][plot_values].std()
                plt.text(hue_order.index(method), ymax-(0.1*ymax), f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')


    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')

    # Save the figure in 300 DPI as a PNG file
    if across == "Resolution":
        save_figure(file_path, f"std_csa_{across.lower()}_{model}.png")
    else:
        if metric == "csa":
            save_figure(file_path, f"std_c2c3_{metric}.png")
        elif metric == "swdice":
            save_figure(file_path, f"{plot_values}_perslice_{metric}.png")


def generate_figure_abs_csa_error(file_path, data, hue_order=None):
    """
    Generate violinplot showing absolute CSA error across participants for each method
    """

    # Remove "softseg_soft" from the list of methods, if it exists
    if 'softseg_soft' in data['Method'].unique():
        data = data[data['Method'] != 'softseg_soft']

    df = pd.DataFrame()
    for method in hue_order[1:]:
        df_error_contrast = pd.DataFrame()
        for contrast in CONTRAST_ORDER:
            df1 = data[(data['Method'] == "softseg_bin") & (data['Contrast'] == contrast)]
            df2 = data[(data['Method'] == method) & (data['Contrast'] == contrast)]

            # group by participant and get the mean area for each participant
            df1 = df1.groupby('Participant')['MEAN(area)'].mean().reset_index()
            df2 = df2.groupby('Participant')['MEAN(area)'].mean().reset_index()

            # compute the absolute error between the two dataframes
            df_temp = pd.merge(df1, df2, on='Participant', suffixes=('_gt', '_contrast'))
            df_error_contrast[contrast] = abs(df_temp['MEAN(area)_gt'] - df_temp['MEAN(area)_contrast'])
        
        df_error_contrast['abs_error_mean'] = df_error_contrast.mean(axis=1)
        df_error_contrast['abs_error_std'] = df_error_contrast.std(axis=1)
        df_error_contrast['Method'] = method
        df_error_contrast['Participant'] = df_temp['Participant']

        df = pd.concat([df, df_error_contrast])
    
    # remove the contrasts from the dataframe
    df = df.drop(columns=CONTRAST_ORDER)
    # # compute the mean and std across contrasts for each method
    # df_agg = df.groupby('Method')[['mean_error', 'std_error']].mean().reset_index()
    # print(df_agg)

    plt.figure(figsize=(12, 6))
    # skip the first method (i.e., softseg_bin)
    sns.violinplot(x='Method', y='abs_error_mean', data=df, order=hue_order)
    # overlay swarm plot on the violin plot to show individual data points
    sns.swarmplot(x='Method', y='abs_error_mean', data=df, color='k', order=hue_order, size=3)

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
        mean = df[df['Method'] == method]['abs_error_mean'].mean()
        std = df[df['Method'] == method]['abs_error_std'].mean()
        plt.text(hue_order.index(method), ymax-1, f'{mean:.2f} +- {std:.2f}', ha='center', va='bottom', color='k')
    
    # Save the figure in 300 DPI as a PNG file
    save_figure(file_path, "abs_csa_error.png")


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
    # Load the CSV file containing averaged (across slices) C2-C3 CSA 
    data_avg_csa = pd.read_csv(args.i)

    # Apply the function to extract participant ID
    data_avg_csa['Participant'] = data_avg_csa['Filename'].apply(fetch_participant_id)

    # Apply the function to extract method and the corresponding analysis details
    if analysis_type == "methods":
        data_avg_csa['Contrast'], data_avg_csa['Method'] = zip(
            *data_avg_csa['Filename'].apply(extract_contrast_and_details, across="Method"))
        
        # Generate violinplot showing STD across participants for each method
        generate_figure_std(data_avg_csa, file_path=args.i, metric="csa")

        if args.i_dice is not None:
            # Generate violinplot showing average slicewise Dice scores across participants for each method

            # Load the CSV file containing averaged slicewise Dice scores
            data_swdice = pd.read_csv(args.i_dice)
            # add the column titles
            data_swdice.columns = ['Filename', 'average_slicewise_dice']

            data_swdice['Participant'] = data_swdice['Filename'].apply(fetch_participant_id)
            data_swdice['Contrast'], data_swdice['Method'] = zip(
                *data_swdice['Filename'].apply(extract_contrast_and_details, across="Method"))

            generate_figure_std(data_swdice, file_path=args.i_dice, metric="swdice")
        else:
            print("No Dice scores provided. Skipping Dice score analysis.")

        # # Generate violinplot showing absolute CSA error across participants for each method
        # generate_figure_abs_csa_error(file_path, data, hue_order=HUE_ORDER)

        # # Generate violinplot showing absolute CSA error for each contrast for a given method
        # for method in HUE_ORDER[1:]:
        #     generate_figure_abs_csa_error_per_contrast(file_path, data, method=method, threshold=None)

    elif analysis_type == "resolutions":
        data_avg_csa['Contrast'], data_avg_csa['Method'], data_avg_csa['Resolution'] = zip(
            *data_avg_csa['Filename'].apply(extract_contrast_and_details, across="Resolution"))
        
        # Generate violinplot showing STD across participants for each resolution
        # generate_figure_std(data, file_path, across="Resolution", hue_order=HUE_ORDER_RES)
        generate_figure_std(data_avg_csa, file_path=args.i, across="Method", hue_order=HUE_ORDER[1:])
    
    else:
        raise ValueError(f'Unknown analysis type: {analysis_type}. Choices: [methods, resolutions, thresholds].')

    # # Compute COV for CSA for each method across resolutions
    # compute_cov(data, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate violin plot from CSV data.')
    parser.add_argument('-i', type=str, 
                        help='Path to the CSV file containing average C2-C3 CSA (to compute STD CSA across contrast)')
    parser.add_argument('-i-dice', type=str, default=None,
                        help="Path to the CSV file containing averaged slice-wise Dice scores for each contrast, method, and subjects")
    parser.add_argument('-a', type=str, default="methods", 
                        help='Options to analyse CSA across. Choices: [methods, resolutions]')
    args = parser.parse_args()
    main(args, analysis_type=args.a)