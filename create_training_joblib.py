import pandas as pd
import joblib
import numpy as np
import argparse
import os


# Inputs:
# --sct_train_file: Pickle file that was holds the a list of the dataset used for training.
#                   Can be downloaded at: https://github.com/sct-data/deepseg_sc_models
#                   train_valid_test column: 1 for training, 2 for validating, 3 for testing
# --bids_datasets_list: List of dataset folders to gather list of subjects from.
#                       1 or more (e.g. sct-testing-large spine-generic-multi-subject etc.)
# --ofolder: Folder to save the output .joblib file

# Example usage:
# python3 create_training_joblib --sct_train_file ~/dataset.pkl --bids_datasets_list ~/datasets/testing-large
#                                --ofolder ~/train_new_model
#
# Konstantinos Nasiotis 2021


def create_new_joblib(dataset_sct_file, input_bids_folders, outputFolder):

    ## Load the merged participants.tsv
    #merged_folder = '/home/nas/Consulting/ivado-project/Datasets/merged_SCTLARGE_MULTISUBJECT/'
    #df_merged = bids.BIDS(merged_folder).participants.content

    # Merge multiple .tsv files into the same dataframe
    df_merged = pd.read_table(os.path.join(input_bids_folders[0], 'participants.tsv'), encoding="ISO-8859-1")
    # Convert to string to get rid of potential TypeError during merging within the same column
    df_merged = df_merged.astype(str)
    # Add the Bids_path to the dataframe
    df_merged['bids_path'] = [input_bids_folders[0]] * len(df_merged)

    for iFolder in range(1, len(input_bids_folders)):
        df_next = pd.read_table(os.path.join(input_bids_folders[iFolder], 'participants.tsv'), encoding="ISO-8859-1")
        df_next = df_next.astype(str)
        df_next['bids_path'] = [input_bids_folders[iFolder]] * len(df_next)
        # Merge the .tsv files (This keeps also non-overlapping fields)
        df_merged = pd.merge(left=df_merged, right=df_next, how='outer')

    dataUsedOnSct = pd.read_pickle(dataset_sct_file)
    # Force the subjects that were used for testing for SCT models to be used for testing in the new .joblib
    subjectsUsedForTesting = dataUsedOnSct[dataUsedOnSct['train_valid_test'] == 3]['subject'].to_list()

    # Use 60% for training/validation and 40% for testing
    percentage_train = 0.4
    percentage_validation = 0.2

    # Whatever was used in sct testing, will stay in the testing side of the joblib as well
    test = df_merged[np.in1d(df_merged['data_id'], subjectsUsedForTesting)]
    # Keep only the rest of the subjects for splitting to training/validation/testing sets
    df_merged_reduced = df_merged[np.invert(np.in1d(df_merged['data_id'], subjectsUsedForTesting))]

    train, validate, test2 = np.split(df_merged_reduced.sample(frac=1),
                                      [int(percentage_train*(len(df_merged_reduced))+len(test)/2),
                                       int((percentage_train+percentage_validation)*len(df_merged_reduced)+len(test)/2)])

    # Append the testing from sct to the new testing entries
    test3 = test.append(test2, ignore_index=1)

    # Populate the joblib file
    jobdict = {'train': train['participant_id'].to_list(),
               'valid': validate['participant_id'].to_list(),
               'test': test3['participant_id'].to_list()}

    joblib.dump(jobdict, os.path.join(outputFolder, "new_splits.joblib"))

    '''
    # Debugging
    newJoblib = joblib.load(os.path.join(outputFolder, "new_splits.joblib"))
    print(len(newJoblib["train"]))
    print(len(newJoblib["valid"]))
    print(len(newJoblib["test"]))
    '''
    print('Success')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sct_train_file", required=True, nargs="*", dest="sctTrainFile",
                        help=".pkl file that was used while training SCT models")
    parser.add_argument("--bids_datasets_list", required=True, nargs="*", dest="bidsDatasets",
                        help="BIDS dataset inputs")
    parser.add_argument("--ofolder", required=True, nargs="*", dest="outputFolder",
                        help="Output folder where the new_splits.joblib file will be saved")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Run comparison
    create_new_joblib(args.sctTrainFile[0], args.bidsDatasets, args.outputFolder[0])


if __name__ == '__main__':
    main()
