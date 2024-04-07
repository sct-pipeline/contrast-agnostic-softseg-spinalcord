import os
import re
import json
import glob
import yaml
from tqdm import tqdm
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import subprocess
from utils import get_git_branch_and_commit
from datetime import datetime

import pandas as pd
pd.set_option('display.max_colwidth', None)


# global variables
# NOTE: each dataset follows a different naming convention with some datasets following 
# an updated BIDS format (e.g., data-multi-subject) and others are in the process of getting updated
# This dict is used to store the names of the folders and corresponding suffixes for each dataset
# convention --> "dataset_name (as per git-annex)": ["labels_folder", "labels_suffix"]
FILESEG_SUFFIXES = {
    "data-multi-subject": ["labels_softseg_bin", "desc-softseg_label-SC_seg"],
    "basel-mp2rage": ["labels_softseg_bin", "desc-softseg_label-SC_seg"],
    "sct-testing-large": ["labels", "seg-manual"],
    "dcm-zurich": ["labels", "seg-manual"],
}


def get_parser():

    parser = argparse.ArgumentParser(description='Code for creating individual datalists for each dataset/contrast for '
                                    'contrast-agnostic SC segmentation.')

    parser.add_argument('--path-data', required=True, type=str, help='Path to BIDS dataset.')
    parser.add_argument('--path-out', type=str, help='Path to the output directory where dataset json is saved')
    # parser.add_argument("--contrast", default="t2w", type=str, help="Contrast to use for training", 
    #                     choices=["t1w", "t2w", "t2star", "mton", "mtoff", "dwi", "all"])
    parser.add_argument('--seed', default=42, type=int, help="Seed for reproducibility")
    # parser.add_argument('--label-type', default='soft_bin', type=str, help="Type of labels to use for training",
    #                     choices=['soft', 'soft_bin'])

    return parser


def get_boilerplate_json(dataset, dataset_commits):
    """
    these are some standard fields that should be included in the json file
    the content of these fields do not really matter, but they should be there only for the sake of consistency
    and so that the MSD datalist loader does not throw an error
    """
    # keys to be defined in the dataset_0.json
    params = OrderedDict()
    params["description"] = "Datasets for contrast-agnostic spinal cord segmentation"
    params["labels"] = {
        "0": "background",
        "1": "sc-seg"
        }
    params["license"] = "MIT"
    params["modality"] = {"0": "MRI"}
    params["dataset"] = dataset
    params["reference"] = "BIDS: Brain Imaging Data Structure"
    params["tensorImageSize"] = "3D"
    params["dataset_versions"] = dataset_commits
    if dataset == 'data-multi-subject':
        params["subject_type"] = "HC"
    elif dataset == 'sct-testing-large':
        params["subject_type"] = ["DCM"]
    
    return params


def fetch_subject_nifti_details(filename_path):
    """
    Get subject ID, session ID and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: subject_session: subject ID and session ID (e.g., sub-001_ses-01) or subject ID (e.g., sub-001)
    Taken from: 
    """

    subject = re.search('sub-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    subjectID = subject.group(0)[:-1] if subject else ""    # [:-1] removes the last underscore or slash

    session = re.search('ses-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    sessionID = session.group(0)[:-1] if session else ""    # [:-1] removes the last underscore or slash

    orientation = re.search('acq-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    orientationID = orientation.group(0)[:-1] if orientation else ""    # [:-1] removes the last underscore or slash

    if 'data-multi-subject' in filename_path:
        # NOTE: the preprocessed spine-generic dataset have a weird BIDS naming convention (due to how they were preprocessed)
        contrast_pattern =  r'.*_(space-other_T1w|space-other_T2w|space-other_T2star|flip-1_mt-on_space-other_MTS|flip-2_mt-off_space-other_MTS|rec-average_dwi).*'
    else:
        # TODO: add more contrasts as needed
        contrast_pattern =  r'.*_(T1w|T2w|T2star|PSIR|STIR|UNIT1|acq-MTon_MTR|acq-dwiMean_dwi).*'
    contrast = re.search(contrast_pattern, filename_path)
    contrastID = contrast.group(1) if contrast else ""

    # REGEX explanation
    # . - match any character (except newline)
    # *? - match the previous element as few times as possible (zero or more times)

    # subject_session = subjectID + '_' + sessionID if subjectID and sessionID else subjectID
    return subjectID, sessionID, orientationID, contrastID


def create_df(dataset_path):
    """
    Create a dataframe with the following columns: subjectID, sessionID, orientationID, age, sex, pathology, notes
    Returns a dataframe with all datasetes merged
    """

    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    labels_folder = FILESEG_SUFFIXES[dataset_name][0]
    labels_suffix = FILESEG_SUFFIXES[dataset_name][1]

    if dataset_name == 'data-multi-subject':
        # get only the (preprocessed) subject files, which are in the `derivatives` folder
        path_files = os.path.join(dataset_path, 'derivatives', 'data_preprocessed', 'sub-*', '**', f'*.nii.gz')
    
    elif dataset_name == 'sct-testing-large':
        
        path_files = os.path.join(dataset_path, 'derivatives', labels_folder, 'sub-*', '**', f'*_{labels_suffix}.nii.gz')

        df_participants = pd.read_csv(os.path.join(dataset_path, 'participants.tsv'), sep='\t')
        # get only those subjects where pathology is DCM
        dcm_subjects = df_participants[df_participants['pathology'] == 'DCM']['participant_id'].tolist()

        # some subjects are in participants.tsv but not in the derivatives/labels folder
        derivatives_subs = os.listdir(os.path.join(dataset_path, 'derivatives', labels_folder))
        dcm_subjects = [sub for sub in dcm_subjects if sub in derivatives_subs]

    else: 
        # fetch the files based on the presence of labels 
        path_files = os.path.join(dataset_path, 'derivatives', labels_folder, 'sub-*', '**', f'*_{labels_suffix}.nii.gz')

    # fetch files only from folders starting with sub-*
    fname_files = glob.glob(path_files, recursive=True)
    if len(fname_files) == 0:
        logger.info(f"No image/label files found in {dataset_path}")
        return None
    
    # create a dataframe with two columns: filesegname and filename
    df = pd.DataFrame({'filename': fname_files})
    df['datasetName'] = os.path.basename(os.path.normpath(dataset_path))
    # get subjectID, sessionID and orientationID
    df['subjectID'], df['sessionID'], df['orientationID'], df['contrastID'] = zip(*df['filename'].map(fetch_subject_nifti_details))

    # if dataset is sct-testing-large, then only include subjects with pathology as DCM
    if dataset_name == 'sct-testing-large':
        df = df[df['subjectID'].isin(dcm_subjects)]

        for file in df['filename']:

            if df['subjectID'].values[0] in dcm_subjects:
                
                # NOTE: sct-testing-large has a lot of images which might/might not have labels. 
                # Get only those images which have labels and are present in the dataframe (and belong to the pathology)
                fname_label = file
                gitannex_cmd_label = f'cd {dataset_path}; git annex get {fname_label}'
                
                fname_image = fname_label.replace(f'/derivatives/{labels_folder}', '').replace(f'_{labels_suffix}.nii.gz', '.nii.gz')
                gitannex_cmd_image = f'cd {dataset_path}; git annex get {fname_image}'

                try:
                    subprocess.run(gitannex_cmd_label, shell=True, check=True)
                    subprocess.run(gitannex_cmd_image, shell=True, check=True)
                    logger.info(f"Downloaded {os.path.basename(fname_label)} from git-annex")
                    logger.info(f"Downloaded {os.path.basename(fname_image)} from git-annex")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error in downloading {file} from git-annex: {e}")
            
            else:
                logger.info(f"Skipping {file} as pathology is not DCM")

    # refactor to move filename and filesegname to the end of the dataframe
    df = df[['datasetName', 'subjectID', 'sessionID', 'orientationID', 'contrastID', 'filename']] #, 'filesegname']]

    return df


def main():

    args = get_parser().parse_args()
    data_root = args.path_data
    timestamp = datetime.now().strftime(f"%Y%m%d-%H%M")  

    # output logger to a file
    logger.add(os.path.join(args.path_out, "logs", f"log_{os.path.basename(data_root)}_seed{args.seed}_{timestamp}.txt"))

    # Check if dataset path exists
    if not os.path.exists(data_root):
        raise ValueError(f"Path {data_root} does not exist.")

    logger.info(f"Creating a dataframe ...")

    # temp dict for storing dataset commits
    dataset_commits = {}

    # create a dataframe for each dataset
    df = create_df(data_root)
        
    # get the git commit ID of the dataset
    dataset_name = os.path.basename(os.path.normpath(data_root))
    branch, commit = get_git_branch_and_commit(data_root)
    dataset_commits[dataset_name] = f"git-{branch}-{commit}"
        
    train_ratio, val_ratio, test_ratio = 0.65, 0.15, 0.2
    # train_subs_all, val_subs_all, test_subs_all = [], [], []
    
    # the idea is to create a datalist for each dataset we want to use 
    # these datalists (which have their own train/val/test splits) for each dataset will then be combined 
    # during the dataloading process of training the contrast-agnostic model

    all_subjects = df['subjectID'].unique()
    train_subjects, test_subjects = train_test_split(all_subjects, test_size=test_ratio, random_state=args.seed)
    # Use the training split to further split into training and validation splits
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_ratio / (train_ratio + val_ratio),
                                                    random_state=args.seed)
    
    # sort the subjects
    train_subjects, val_subjects, test_subjects = sorted(train_subjects), sorted(val_subjects), sorted(test_subjects)

    # get boilerplate json
    params = get_boilerplate_json(dataset_name, dataset_commits)

    train_subjects_dict = {"train": train_subjects}
    val_subjects_dict = {"validation": val_subjects}
    test_subjects_dict =  {"test": test_subjects}
    all_subjects_list = [train_subjects_dict, val_subjects_dict, test_subjects_dict]

    # list of subjects whose labels don't exist (only for spine-generic)
    subjects_to_remove = []

    labels_folder = FILESEG_SUFFIXES[dataset_name][0]
    labels_suffix = FILESEG_SUFFIXES[dataset_name][1]

    # iterate through train and test subjects
    for subjects_dict in tqdm(all_subjects_list, desc="Iterating through train/val/test splits"):

        for name, subs_list in subjects_dict.items():
            
            temp_list = []            
            for subject_no, subject in enumerate(subs_list):

                # get all the contrastIDs for the subject
                contrastIDs = df[df['subjectID'] == subject]['contrastID'].unique()

                for contrast in contrastIDs:
                
                    temp_data = {}
                    # if the subject belongs to a data-multi-subject dataset, then the filename is different
                    if df['datasetName'].values[0] == 'data-multi-subject':
                        # NOTE: for spine-generic subjects, we're pulling the data from image filename
                        fname_image = df[(df['subjectID'] == subject) & (df['contrastID'] == contrast)]['filename'].values[0]
                        fname_label = fname_image.replace('data_preprocessed', labels_folder).replace('.nii.gz', f'_{labels_suffix}.nii.gz')
                    
                    else: 
                        # NOTE: but for other datasets, we are getting them from the lesion filenames
                        fname_label = df[(df['subjectID'] == subject) & (df['contrastID'] == contrast)]['filename'].values[0]
                        fname_image = fname_label.replace(f'/derivatives/{labels_folder}', '').replace(f'_{labels_suffix}.nii.gz', '.nii.gz')
                                    
                    temp_data["image"] = fname_image
                    temp_data["label"] = fname_label

                    if os.path.exists(temp_data["image"]) and os.path.exists(temp_data["label"]):
                        temp_list.append(temp_data)
                    else:
                        if not os.path.exists(temp_data["image"]):
                            logger.info(f"{temp_data['image']} does not exist.")
                        if not os.path.exists(temp_data["label"]):
                            logger.info(f"{temp_data['label']} does not exist.")
                        subjects_to_remove.append(subject)

                params[name] = temp_list

    # log the contrasts used
    params["contrasts"] = df['contrastID'].unique().tolist()

    # number of training, validation and testing images (not subjects; a subject can have multiple contrasts, and hence multiple images)
    params["numTrainingImages"] = len(params["train"])
    params["numValidationImages"] = len(params["validation"])
    params["numTestImages"] = len(params["test"])
    params["seed"] = args.seed

    # update the number of train/val/test subjects
    train_subs_all = list(set(train_subjects) - set(subjects_to_remove))
    val_subs_all = list(set(val_subjects) - set(subjects_to_remove))
    test_subs_all = list(set(test_subjects) - set(subjects_to_remove))
    params["numTrainingSubjects"] = len(train_subs_all)
    params["numValidationSubjects"] = len(val_subs_all)
    params["numTestSubjects"] = len(test_subs_all)

    logger.info(f"Number of training images (not subjects): {params['numTrainingImages']}")
    logger.info(f"Number of validation images (not subjects): {params['numValidationImages']}")
    logger.info(f"Number of testing images (not subjects): {params['numTestImages']}")

    # dump train/val/test splits into a yaml file
    with open(f"datasplit_{dataset_name}_seed{args.seed}.yaml", 'w') as file:
        yaml.dump({'train': sorted(train_subs_all), 'val': sorted(val_subs_all), 'test': sorted(test_subs_all)}, file, indent=2, sort_keys=True)

    final_json = json.dumps(params, indent=4, sort_keys=True)
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out, exist_ok=True)

    # jsonFile = open(args.path_out + "/" + f"dataset_{contrast}_{args.label_type}_seed{seed}.json", "w")
    jsonFile = open(args.path_out + "/" + f"datasplit_{dataset_name}_seed{args.seed}_{timestamp}.json", "w")
    jsonFile.write(final_json)
    jsonFile.close()


if __name__ == "__main__":
    main()
