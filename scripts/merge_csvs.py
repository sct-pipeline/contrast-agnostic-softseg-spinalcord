"""
Merge multiple CSV files into a single CSV file.
"""

import argparse
import os
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-path-results', help='Path to the sct_run_batch output folder containing the results/ folder')
    parser.add_argument('-path-output', help='Path to the output folder where the merged CSV file will be saved')
    # parser.add_argument('output_csv', help='Output CSV file')
    return parser


def merge_csvs(args):

    # Get list of CSV files
    list_csv = []

    # 49 test subjects for 15 batches with 3 subjects each, 16th batch with 4 subjects
    max_batches=16
    for idx in range(1, max_batches+1):
        list_csv.append(os.path.join(args.path_results, f'csa-results-batch-{idx}', 'results', 'csa_c2c3.csv'))

    # Merge CSV files
    df = pd.concat([pd.read_csv(f) for f in list_csv], ignore_index=True)

    path_save = os.path.join(args.path_output, 'csa_c2c3_merged.csv')
    print(f'Saving merged CSV file to: {path_save}')
    df.to_csv(path_save, index=False)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    merge_csvs(args)