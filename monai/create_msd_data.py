import os
import json
from tqdm import tqdm
import yaml
import argparse
import joblib
from utils import FoldGenerator
from loguru import logger
from sklearn.model_selection import train_test_split

# root = "/home/GRAMES.POLYMTL.CA/u114716/datasets/spine-generic_uncropped"

parser = argparse.ArgumentParser(description='Code for MSD-style JSON datalist for spine-generic dataset.')

parser.add_argument('-pd', '--path-data', required=True, type=str, help='Path to the data set directory')
parser.add_argument('-pj', '--path-joblib', help='Path to joblib file from ivadomed containing the dataset splits.',
                    default=None, type=str)
parser.add_argument('-po', '--path-out', type=str, help='Path to the output directory where dataset json is saved')
parser.add_argument("--contrast", default="t2w", type=str, help="Contrast to use for training", 
                    choices=["t1w", "t2w", "t2star", "mton", "mtoff", "dwi", "all"])
parser.add_argument('--label-type', default='soft', type=str, help="Type of labels to use for training",
                    choices=['hard', 'soft'])
parser.add_argument('--seed', default=42, type=int, help="Seed for reproducibility")
args = parser.parse_args()


root = args.path_data
seed = args.seed
contrast = args.contrast
if args.label_type == 'soft':
    logger.info("Using SOFT LABELS ...")
    PATH_DERIVATIVES = os.path.join(root, "derivatives", "labels_softseg")
    SUFFIX = "softseg"
else:
    logger.info("Using HARD LABELS ...")
    PATH_DERIVATIVES = os.path.join(root, "derivatives", "labels")
    SUFFIX = "seg-manual"

# Get all subjects
# the participants.tsv file might not be up-to-date, hence rely on the existing folders
# subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
# subjects = subjects_df['participant_id'].values.tolist()
subjects = [subject for subject in os.listdir(root) if subject.startswith('sub-')]
logger.info(f"Total number of subjects in the root directory: {len(subjects)}")

if args.path_joblib is not None:
    # load information from the joblib to match train and test subjects
    # joblib_file = os.path.join(args.path_joblib, 'split_datasets_all_seed=15.joblib')
    splits = joblib.load(args.path_joblib)
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
    # sort the subjects
    train_subjects = sorted(train_subjects)
    val_subjects = sorted(val_subjects)
    test_subjects = sorted(test_subjects)

logger.info(f"Number of training subjects: {len(train_subjects)}")
logger.info(f"Number of validation subjects: {len(val_subjects)}")
logger.info(f"Number of testing subjects: {len(test_subjects)}")

# dump train/val/test splits into a yaml file
with open(f"data_split_{contrast}_{args.label_type}_seed{seed}.yaml", 'w') as file:
    yaml.dump({'train': train_subjects, 'val': val_subjects, 'test': test_subjects}, file, indent=2, sort_keys=True)

# keys to be defined in the dataset_0.json
params = {}
params["description"] = "spine-generic-uncropped"
params["labels"] = {
    "0": "background",
    "1": "soft-sc-seg"
    }
params["license"] = "nk"
params["modality"] = {
    "0": "MRI"
    }
params["name"] = "spine-generic"
params["numTest"] = len(test_subjects)
params["numTraining"] = len(train_subjects)
params["numValidation"] = len(val_subjects)
params["seed"] = args.seed
params["reference"] = "University of Zurich"
params["tensorImageSize"] = "3D"

train_subjects_dict = {"train": train_subjects}
val_subjects_dict = {"validation": val_subjects}
test_subjects_dict =  {"test": test_subjects}
all_subjects_list = [train_subjects_dict, val_subjects_dict, test_subjects_dict]

# # define the contrasts
# contrasts_list = ['T1w', 'T2w', 'T2star', 'flip-1_mt-on_MTS', 'flip-2_mt-off_MTS', 'dwi']

for subjects_dict in tqdm(all_subjects_list, desc="Iterating through train/val/test splits"):

    for name, subs_list in subjects_dict.items():

        temp_list = []
        for subject_no, subject in enumerate(subs_list):

            if contrast == "all":
                temp_data_t1w = {}
                temp_data_t2w = {}
                temp_data_t2star = {}
                temp_data_mton_mts = {}
                temp_data_mtoff_mts = {}
                temp_data_dwi = {}

                # t1w
                temp_data_t1w["image"] = os.path.join(root, subject, 'anat', f"{subject}_T1w.nii.gz")
                temp_data_t1w["label"] = os.path.join(PATH_DERIVATIVES, subject, 'anat', f"{subject}_T1w_{SUFFIX}.nii.gz")
                if os.path.exists(temp_data_t1w["label"]) and os.path.exists(temp_data_t1w["image"]):
                    temp_list.append(temp_data_t1w)

                # t2w
                temp_data_t2w["image"] = os.path.join(root, subject, 'anat', f"{subject}_T2w.nii.gz")
                temp_data_t2w["label"] = os.path.join(PATH_DERIVATIVES, subject, 'anat', f"{subject}_T2w_{SUFFIX}.nii.gz")
                if os.path.exists(temp_data_t2w["label"]) and os.path.exists(temp_data_t2w["image"]):
                    temp_list.append(temp_data_t2w)

                # t2star
                temp_data_t2star["image"] = os.path.join(root, subject, 'anat', f"{subject}_T2star.nii.gz")
                temp_data_t2star["label"] = os.path.join(PATH_DERIVATIVES, subject, 'anat', f"{subject}_T2star_{SUFFIX}.nii.gz")
                if os.path.exists(temp_data_t2star["label"]) and os.path.exists(temp_data_t2star["image"]):
                    temp_list.append(temp_data_t2star)

                # mton_mts
                temp_data_mton_mts["image"] = os.path.join(root, subject, 'anat', f"{subject}_flip-1_mt-on_MTS.nii.gz")
                temp_data_mton_mts["label"] = os.path.join(PATH_DERIVATIVES, subject, 'anat', f"{subject}_flip-1_mt-on_MTS_{SUFFIX}.nii.gz")
                if os.path.exists(temp_data_mton_mts["label"]) and os.path.exists(temp_data_mton_mts["image"]):
                    temp_list.append(temp_data_mton_mts)

                # t1w_mts
                temp_data_mtoff_mts["image"] = os.path.join(root, subject, 'anat', f"{subject}_flip-2_mt-off_MTS.nii.gz")
                temp_data_mtoff_mts["label"] = os.path.join(PATH_DERIVATIVES, subject, 'anat', f"{subject}_flip-2_mt-off_MTS_{SUFFIX}.nii.gz")
                if os.path.exists(temp_data_mtoff_mts["label"]) and os.path.exists(temp_data_mtoff_mts["image"]):
                    temp_list.append(temp_data_mtoff_mts)

                # dwi
                temp_data_dwi["image"] = os.path.join(root, subject, 'dwi', f"{subject}_rec-average_dwi.nii.gz")
                temp_data_dwi["label"] = os.path.join(PATH_DERIVATIVES, subject, 'dwi', f"{subject}_rec-average_dwi_{SUFFIX}.nii.gz")
                if os.path.exists(temp_data_dwi["label"]) and os.path.exists(temp_data_dwi["image"]):
                    temp_list.append(temp_data_dwi)


            elif contrast == "t1w":     # t1w
                temp_data_t1w = {}
                temp_data_t1w["image"] = os.path.join(root, subject, 'anat', f"{subject}_T1w.nii.gz")
                temp_data_t1w["label"] = os.path.join(PATH_DERIVATIVES, subject, 'anat', f"{subject}_T1w_{SUFFIX}.nii.gz")
                if os.path.exists(temp_data_t1w["label"]) and os.path.exists(temp_data_t1w["image"]):
                    temp_list.append(temp_data_t1w)
                else:
                    logger.info(f"Subject {subject} does not have T1w image or label.")


            elif contrast == "t2w":     # t2w
                temp_data_t2w = {}
                temp_data_t2w["image"] = os.path.join(root, subject, 'anat', f"{subject}_T2w.nii.gz")
                temp_data_t2w["label"] = os.path.join(PATH_DERIVATIVES, subject, 'anat', f"{subject}_T2w_{SUFFIX}.nii.gz")
                if os.path.exists(temp_data_t2w["label"]) and os.path.exists(temp_data_t2w["image"]):
                    temp_list.append(temp_data_t2w)
                else:
                    logger.info(f"Subject {subject} does not have T2w image or label.")


            elif contrast == "t2star":     # t2star
                temp_data_t2star = {}
                temp_data_t2star["image"] = os.path.join(root, subject, 'anat', f"{subject}_T2star.nii.gz")
                temp_data_t2star["label"] = os.path.join(PATH_DERIVATIVES, subject, 'anat', f"{subject}_T2star_{SUFFIX}.nii.gz")
                if os.path.exists(temp_data_t2star["label"]) and os.path.exists(temp_data_t2star["image"]):
                    temp_list.append(temp_data_t2star)
                else:
                    logger.info(f"Subject {subject} does not have T2star image or label.")


            elif contrast == "mton":     # mton_mts
                temp_data_mton_mts = {}
                temp_data_mton_mts["image"] = os.path.join(root, subject, 'anat', f"{subject}_flip-1_mt-on_MTS.nii.gz")
                temp_data_mton_mts["label"] = os.path.join(PATH_DERIVATIVES, subject, 'anat', f"{subject}_flip-1_mt-on_MTS_{SUFFIX}.nii.gz")
                if os.path.exists(temp_data_mton_mts["label"]) and os.path.exists(temp_data_mton_mts["image"]):
                    temp_list.append(temp_data_mton_mts)
                else:
                    logger.info(f"Subject {subject} does not have MTOn image or label.")

            elif contrast == "mtoff":     # t1w_mts
                temp_data_mtoff_mts = {}
                temp_data_mtoff_mts["image"] = os.path.join(root, subject, 'anat', f"{subject}_flip-2_mt-off_MTS.nii.gz")
                temp_data_mtoff_mts["label"] = os.path.join(PATH_DERIVATIVES, subject, 'anat', f"{subject}_flip-2_mt-off_MTS_{SUFFIX}.nii.gz")
                if os.path.exists(temp_data_mtoff_mts["label"]) and os.path.exists(temp_data_mtoff_mts["image"]):
                    temp_list.append(temp_data_mtoff_mts)
                else:
                    logger.info(f"Subject {subject} does not have MTOff image or label.")

            elif contrast == "dwi":     # dwi
                temp_data_dwi = {}
                temp_data_dwi["image"] = os.path.join(root, subject, 'dwi', f"{subject}_rec-average_dwi.nii.gz")
                temp_data_dwi["label"] = os.path.join(PATH_DERIVATIVES, subject, 'dwi', f"{subject}_rec-average_dwi_{SUFFIX}.nii.gz")
                if os.path.exists(temp_data_dwi["label"]) and os.path.exists(temp_data_dwi["image"]):
                    temp_list.append(temp_data_dwi)
                else:
                    logger.info(f"Subject {subject} does not have DWI image or label.")

            else:
                raise ValueError(f"Contrast {contrast} not recognized.")
            
        
        params[name] = temp_list
        logger.info(f"Number of images in {name} set: {len(temp_list)}")

final_json = json.dumps(params, indent=4, sort_keys=True)
if not os.path.exists(args.path_out):
    os.makedirs(args.path_out, exist_ok=True)

jsonFile = open(args.path_out + "/" + f"dataset_{contrast}_{args.label_type}_seed{seed}.json", "w")
jsonFile.write(final_json)
jsonFile.close()



    


