"""
Loop across csv files containing sequence parameters (one for each dataset) and combine the averaged details 
into a single csv file.:

Assumes that the input folder contains the CSV file (parsed_data.csv) for datasets included in the training of 
the contrast-agnostic model.

Example usage:
    python scripts/combine_dataset_stats.py -i /path/to/folder/sequence_parameters 

Author: Naga Karthik
    
"""

import os
import re
import glob
import json
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import nibabel as nib


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Loop across CSV files in the input folder and parse from them relevant information.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i', required=True, type=str,
        help='Path to the folder containing the CSV files containing the sequence parameters for the datasets. ' 
             'typically the output of fetch_sequence_parameters.py script.'
    )

    return parser

def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    list_of_csvs = glob.glob(os.path.join(args.i, '*.csv'), recursive=True)

    # Create a pandas DataFrame from the parsed data
    df = pd.DataFrame()
    for csv_file in list_of_csvs:
        temp_dict = {}
        
        csv_path = os.path.join(args.i, csv_file)
        print(f"Processing {csv_path}")
        df_temp = pd.read_csv(csv_path)
        
        dataset_name = csv_path.split('/')[-1].split('_')[0]
        # Add the dataset name to the new dataframe
        temp_dict['Dataset'] = dataset_name

        # get the unique values of the contrast
        temp_dict['Contrast'] = list(df_temp['Contrast'].unique())
        
        # get the unique values of "MagneticFieldStrength"
        scanners = []
        scanner_strengths = df_temp['MagneticFieldStrength'].unique()
        # remove nan values
        scanner_strengths = [x for x in scanner_strengths if str(x) != 'nan']
        for scanner in scanner_strengths:
            scanners.append(f"{scanner}T")
        temp_dict['Scanners'] = scanners

        # get the unique values of "Manufacturer"
        temp_dict['Manufacturer'] = list(df_temp['Manufacturer'].unique())

        # get the min-max ranges for PixDim along each dimension
        temp_dict['PixDim_min'] = df_temp['PixDim'].min()
        temp_dict['PixDim_max'] = df_temp['PixDim'].max()

        # get the min-max ranges for SliceThickness
        temp_dict['SliceThickness_min'] = df_temp['SliceThickness'].min()
        temp_dict['SliceThickness_max'] = df_temp['SliceThickness'].max()

        temp_dict['Authors'] = 'n/a'

        # add the dictionary to the dataframe
        df_dataset = pd.DataFrame.from_dict(temp_dict, orient='index').T

        df = pd.concat([df, df_dataset])


    # Save the DataFrame to a CSV file
    out_path = os.path.join(args.i, 'combined_datasets_stats.csv')
    df.to_csv(out_path, index=False)
    

if __name__ == "__main__":
    main()
