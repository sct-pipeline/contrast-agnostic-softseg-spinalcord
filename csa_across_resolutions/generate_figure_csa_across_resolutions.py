"""
Generate violin plot from CSV data across resolutions and methods.

Usage:
    python generate_figure_csa_across_resolutions.py -i /path/to/data.csv
"""

import argparse
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt


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


def main(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Apply the function to extract method and resolution
    data['Method'], data['Resolution'] = zip(*data['Filename'].apply(extract_method_resolution))

    # Correct labels for the x-axis based on the actual data
    resolution_labels = ['1mm', '125mm', '15mm', '175mm', '2mm']

    # Setting the hue order as specified
    hue_order = ["propseg", "deepseg_2d", "nnunet_3d_fullres", "monai"]

    # Creating the violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Resolution', y='MEAN(area)', hue='Method', data=data, order=resolution_labels,
                   hue_order=hue_order)
    plt.xticks(rotation=45)
    plt.xlabel('Resolution')
    plt.ylabel('CSA [mm^2]')
    plt.title('CSA across Resolutions and Methods')
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Add horizontal dashed grid
    plt.grid(axis='y', alpha=0.5, linestyle='dashed')

    # Update x-axis labels
    plt.gca().set_xticklabels(['1mm', '1.25mm', '1.5mm', '1.75mm', '2mm'])

    # Save the figure in 300 DPI as a PNG file
    plt.savefig(file_path.replace('.csv', '.png'), dpi=300)
    print(f'Figure saved to {file_path.replace(".csv", ".png")}')

    # Display the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate violin plot from CSV data.')
    parser.add_argument('-i', type=str, help='Path to the CSV file')
    args = parser.parse_args()
    main(args.i)
