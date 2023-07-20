import os
import json
from tqdm import tqdm
import numpy as np
import argparse
import joblib
from utils import FoldGenerator
from loguru import logger
from sklearn.model_selection import train_test_split

# For now this script will always use the joblib test set from spine generic for comparison. All other datasets are directly sent to the training set.

# TODO: Add loop for multiple datasets only to add in training
# TODO: Edit loop to custom train and test set from the datasets

root = "/home/GRAMES.POLYMTL.CA/lobouz/duke/projects/ivadomed/contrast-agnostic-seg/data_processed_sg_2023-03-10_NO_CROP/data_processed_clean"

parser = argparse.ArgumentParser(description='Code for creating k-fold splits of the spine-generic dataset.')

parser.add_argument('--seed', default=15, type=int, help="Seed for reproducibility")
parser.add_argument('-ncvf', '--num-cv-folds', default=0, type=int, 
            help="[1-k] To create a k-fold dataset for cross validation, 0 for single file with all subjects")
parser.add_argument('-pd', '--path-data-SG', default=root, type=str, help='Path to the data set directory')
parser.add_argument('-pj', '--path-joblib', help='Path to joblib file from ivadomed containing the dataset splits.',
                    default=None, type=str)
parser.add_argument('-po', '--path-out', default="/home/GRAMES.POLYMTL.CA/lobouz/data_tmp", type=str, help='Path to the output directory where dataset json is saved')
parser.add_argument('-dp', "--datasets-paths", nargs="*", help="List of paths to all the datasets to aggregate in the JSON.")
parser.add_argument('-dn', '--dataset-name-v', default="aggregated_dataset_v0", type=str, help='Name of the dataset built with version.')
parser.add_argument('-dd', '--dataset-description', default="Aggregated dataset", type=str, help='Description of the dataset built (ideally all dataset names).')
args = parser.parse_args()


root = args.path_data_SG
seed = args.seed
num_cv_folds = args.num_cv_folds    # for 100 subjects, performs a 60-20-20 split with num_cv_folds

# Get all subjects
# the participants.tsv file might not be up-to-date, hence rely on the existing folders
# subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
# subjects = subjects_df['participant_id'].values.tolist()
subjects = [subject for subject in os.listdir(root) if subject.startswith('sub-')]
logger.info(f"Total number of subjects in the root directory: {len(subjects)}")

if args.num_cv_folds != 0:
    # create k-fold CV datasets as usual
    
    # returns a nested list of length (num_cv_folds), each element (again, a list) consisting of 
    # train, val, test indices and the fold number
    names_list = FoldGenerator(seed, num_cv_folds, len_data=len(subjects)).get_fold_names()

    for fold in range(num_cv_folds):

        train_ix, val_ix, test_ix, fold_num = names_list[fold]
        training_subjects = [subjects[tr_ix] for tr_ix in train_ix]
        validation_subjects = [subjects[v_ix] for v_ix in val_ix]
        test_subjects = [subjects[te_ix] for te_ix in test_ix]

        # keys to be defined in the dataset_0.json
        params = {}
        params["description"] = "sci-zurich naga"
        params["labels"] = {
            "0": "background",
            "1": "sc-lesion"
            }
        params["license"] = "nk"
        params["modality"] = {
            "0": "MRI"
            }
        params["name"] = "sci-zurich"
        params["numTest"] = len(test_subjects)
        params["numTraining"] = len(training_subjects) + len(validation_subjects)
        params["reference"] = "University of Zurich"
        params["tensorImageSize"] = "3D"


        train_val_subjects_dict = {
            "training": training_subjects,
            "validation": validation_subjects,
        } 
        test_subjects_dict =  {"test": test_subjects}

        # run loop for training and validation subjects
        temp_shapes_list = []
        for name, subs_list in train_val_subjects_dict.items():

            temp_list = []
            for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
            
                # Another for loop for going through sessions
                temp_subject_path = os.path.join(root, subject)
                num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

                for ses_idx in range(1, num_sessions_per_subject+1):
                    temp_data = {}
                    # Get paths with session numbers
                    session = 'ses-0' + str(ses_idx)
                    subject_images_path = os.path.join(root, subject, session, 'anat')
                    subject_labels_path = os.path.join(root, 'derivatives', 'labels', subject, session, 'anat')

                    subject_image_file = os.path.join(subject_images_path, '%s_%s_acq-sag_T2w.nii.gz' % (subject, session))
                    subject_label_file = os.path.join(subject_labels_path, '%s_%s_acq-sag_T2w_lesion-manual.nii.gz' % (subject, session))

                    # get shapes of each subject to calculate median later
                    # temp_shapes_list.append(np.shape(nib.load(subject_image_file).get_fdata()))

                    # # load GT mask
                    # gt_label = nib.load(subject_label_file).get_fdata()
                    # bbox_coords = get_bounding_boxes(mask=gt_label)

                    # store in a temp dictionary
                    temp_data["image"] = subject_image_file.replace(root+"/", '') # .strip(root)
                    temp_data["label"] = subject_label_file.replace(root+"/", '') # .strip(root)
                    # temp_data["box"] = bbox_coords
                    
                    temp_list.append(temp_data)
            
            params[name] = temp_list

            # print(temp_shapes_list)
            # calculate the median shapes along each axis
            params["train_val_median_shape"] = np.median(temp_shapes_list, axis=0).tolist()

        # run separate loop for testing 
        for name, subs_list in test_subjects_dict.items():
            temp_list = []
            for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
            
                # Another for loop for going through sessions
                temp_subject_path = os.path.join(root, subject)
                num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))

                for ses_idx in range(1, num_sessions_per_subject+1):
                    temp_data = {}
                    # Get paths with session numbers
                    session = 'ses-0' + str(ses_idx)
                    subject_images_path = os.path.join(root, subject, session, 'anat')
                    subject_labels_path = os.path.join(root, 'derivatives', 'labels', subject, session, 'anat')
                    
                    subject_image_file = os.path.join(subject_images_path, '%s_%s_acq-sag_T2w.nii.gz' % (subject, session))
                    subject_label_file = os.path.join(subject_labels_path, '%s_%s_acq-sag_T2w_lesion-manual.nii.gz' % (subject, session))

                    # # load GT mask
                    # gt_label = nib.load(subject_label_file).get_fdata()
                    # bbox_coords = get_bounding_boxes(mask=gt_label)

                    temp_data["image"] = subject_image_file.replace(root+"/", '')
                    temp_data["label"] = subject_label_file.replace(root+"/", '')
                    # temp_data["box"] = bbox_coords

                    temp_list.append(temp_data)
            
            params[name] = temp_list

        final_json = json.dumps(params, indent=4, sort_keys=True)
        jsonFile = open(root + "/" + f"dataset_fold-{fold_num}.json", "w")
        jsonFile.write(final_json)
        jsonFile.close()
else: 

    if args.path_joblib is not None:
        # load information from the joblib to match train and test subjects
        joblib_file = os.path.join(args.path_joblib, 'split_datasets_all_seed=15.joblib')
        splits = joblib.load("split_datasets_all_seed=15.joblib")
        # get the subjects from the joblib file
        train_subjects = sorted(list(set([sub.split('_')[0] for sub in splits['train']])))
        val_subjects = sorted(list(set([sub.split('_')[0] for sub in splits['valid']])))
        test_subjects = sorted(list(set([sub.split('_')[0] for sub in splits['test']])))

    else:
        # create one json file with 60-20-20 train-val-test split
        train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
        train_subjects, test_subjects = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)
        # Use the training split to further split into training and validation splits
        train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_ratio / (train_ratio + val_ratio),
                                                        random_state=args.seed, )

    logger.info(f"(Prior aggregation) Number of training subjects: {len(train_subjects)}")
    logger.info(f"(Prior aggregation) Number of validation subjects: {len(val_subjects)}")
    logger.info(f"(Prior aggregation) Number of testing subjects: {len(test_subjects)}")

    # Add train and val data with 80:20 split from all other datasets
    for idx, dataset_to_add in enumerate(args.datasets_paths):
        # Get the subjects
        subjects_to_add = [subject for subject in os.listdir(dataset_to_add) if subject.startswith('sub-')]
        logger.info(f"Total number of subjects in the dataset {idx} directory: {len(subjects_to_add)}")

        # Aggregate in val and train with 80:20 split
        train_ratio, val_ratio = 0.8, 0.2
        train_subjects_tmp, val_subjects_tmp = train_test_split(subjects_to_add, test_size=val_ratio / (train_ratio + val_ratio), random_state=args.seed)
        train_subjects.extend(train_subjects_tmp)
        val_subjects.extend(val_subjects_tmp)

    logger.info(f"(Post aggregation) Number of training subjects: {len(train_subjects)}")
    logger.info(f"(Post aggregation) Number of validation subjects: {len(val_subjects)}")
    logger.info(f"(Post aggregation) Number of testing subjects: {len(test_subjects)}")

    # keys to be defined in the dataset_0.json
    params = {}
    params["description"] = args.dataset_description
    params["labels"] = {
        "0": "background",
        "1": "soft-sc-seg"
        }
    params["license"] = "nk"
    params["modality"] = {
        "0": "MRI"
        }
    params["name"] = args.dataset_name_v
    params["numTest"] = len(test_subjects)
    params["numTraining"] = len(train_subjects)
    params["numValidation"] = len(val_subjects)
    params["seed"] = args.seed
    params["reference"] = "TBD"
    params["tensorImageSize"] = "3D"

    train_subjects_dict = {"train": train_subjects}
    val_subjects_dict = {"validation": val_subjects}
    test_subjects_dict =  {"test": test_subjects}
    all_subjects_list = [train_subjects_dict, val_subjects_dict, test_subjects_dict]


    # define the variables for iteration
    contrasts_list = ['T1w', 'T2w', 'T2star', 'flip-1_mt-on_MTS', 'flip-2_mt-off_MTS', 'dwi', 'UNIT1','acq-cspineAxial_T2w', 'task-motor_bold',
                      'task-thermal_bold', 'task-rest_bold', 'task-BilatMotorThermal_bold', 'task-tactile_bold']
    label_type_list = ['label-SC_seg', 'seg-manual']
    label_names = ['labels','manual_labels']
    anat_types = ['anat', 'func']
    for subjects_dict in tqdm(all_subjects_list, desc="Iterating through train/val/test splits"):

        for name, subs_list in subjects_dict.items():

            temp_list = []
            for subject_no, subject in enumerate(subs_list):

                temp_data_t1w = {}
                temp_data_t2w = {}
                temp_data_t2star = {}
                temp_data_mton_mts = {}
                temp_data_mtoff_mts = {}
                temp_data_dwi = {}
                temp_data_UNIT1 = {}

                # t1w
                temp_data_t1w["image"] = os.path.join(root, subject, 'anat', f"{subject}_T1w.nii.gz")
                temp_data_t1w["label"] = os.path.join(root, "derivatives", "labels_softseg", subject, 'anat', f"{subject}_T1w_softseg.nii.gz")
                if os.path.exists(temp_data_t1w["label"]) and os.path.exists(temp_data_t1w["image"]):
                    temp_list.append(temp_data_t1w)

                # t2w
                temp_data_t2w["image"] = os.path.join(root, subject, 'anat', f"{subject}_T2w.nii.gz")
                temp_data_t2w["label"] = os.path.join(root, "derivatives", "labels_softseg", subject, 'anat', f"{subject}_T2w_softseg.nii.gz")
                if os.path.exists(temp_data_t2w["label"]) and os.path.exists(temp_data_t2w["image"]):
                    temp_list.append(temp_data_t2w)

                # t2star
                temp_data_t2star["image"] = os.path.join(root, subject, 'anat', f"{subject}_T2star.nii.gz")
                temp_data_t2star["label"] = os.path.join(root, "derivatives", "labels_softseg", subject, 'anat', f"{subject}_T2star_softseg.nii.gz")
                if os.path.exists(temp_data_t2star["label"]) and os.path.exists(temp_data_t2star["image"]):
                    temp_list.append(temp_data_t2star)

                # mton_mts
                temp_data_mton_mts["image"] = os.path.join(root, subject, 'anat', f"{subject}_flip-1_mt-on_MTS.nii.gz")
                temp_data_mton_mts["label"] = os.path.join(root, "derivatives", "labels_softseg", subject, 'anat', f"{subject}_flip-1_mt-on_MTS_softseg.nii.gz")
                if os.path.exists(temp_data_mton_mts["label"]) and os.path.exists(temp_data_mton_mts["image"]):
                    temp_list.append(temp_data_mton_mts)

                # t1w_mts
                temp_data_mtoff_mts["image"] = os.path.join(root, subject, 'anat', f"{subject}_flip-2_mt-off_MTS.nii.gz")
                temp_data_mtoff_mts["label"] = os.path.join(root, "derivatives", "labels_softseg", subject, 'anat', f"{subject}_flip-2_mt-off_MTS_softseg.nii.gz")
                if os.path.exists(temp_data_mtoff_mts["label"]) and os.path.exists(temp_data_mtoff_mts["image"]):
                    temp_list.append(temp_data_mtoff_mts)

                # dwi
                temp_data_dwi["image"] = os.path.join(root, subject, 'dwi', f"{subject}_rec-average_dwi.nii.gz")
                temp_data_dwi["label"] = os.path.join(root, "derivatives", "labels_softseg", subject, 'dwi', f"{subject}_rec-average_dwi_softseg.nii.gz")
                if os.path.exists(temp_data_dwi["label"]) and os.path.exists(temp_data_dwi["image"]):
                    temp_list.append(temp_data_dwi)

                # Loop through all other datasets
                # This loop currently works with (confirmed): Basel-MP2RAGE, inspired, sci-colorado, canproco
                for root_others in args.datasets_paths:
                    # loop over contrasts
                    for contrast in contrasts_list:
                        # Loop over different label types for SC seg
                        for label_type in label_type_list:
                            # Loop over label names
                            for label_name in label_names:
                                # Loop over anat types
                                for anat_type in anat_types:
                                    temp_data_UNIT1["image"] = os.path.join(root_others, subject, anat_type, f"{subject}_{contrast}.nii.gz")
                                    temp_data_UNIT1["label"] = os.path.join(root_others, "derivatives", label_name, subject, anat_type, f"{subject}_{contrast}_{label_type}.nii.gz")
                                    if os.path.exists(temp_data_UNIT1["label"]) and os.path.exists(temp_data_UNIT1["image"]):
                                        temp_list.append(temp_data_UNIT1)

                                    # Manual addition for session
                                    temp_data_UNIT1["image"] = os.path.join(root_others, subject, 'ses-M0', anat_type, f"{subject}_ses-M0_{contrast}.nii.gz")
                                    temp_data_UNIT1["label"] = os.path.join(root_others, "derivatives", label_name, subject, 'ses-M0', anat_type, f"{subject}_ses-M0_{contrast}_{label_type}.nii.gz")
                                    if os.path.exists(temp_data_UNIT1["label"]) and os.path.exists(temp_data_UNIT1["image"]):
                                        temp_list.append(temp_data_UNIT1)
            
            params[name] = temp_list
            logger.info(f"Number of images in {name} set: {len(temp_list)}")

    final_json = json.dumps(params, indent=4, sort_keys=True)
    jsonFile = open(args.path_out + "/" + f"dataset.json", "w")
    jsonFile.write(final_json)
    jsonFile.close()



    