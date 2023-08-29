import os
import json
import argparse
import joblib
from loguru import logger

parser = argparse.ArgumentParser(description='Code for creating k-fold splits of the spine-generic dataset.')

parser.add_argument('-dname', '--dataset-name', default='spine-generic', type=str, help='Name of the dataset')
parser.add_argument('-pd', '--path-data', required=True, type=str, help='Path to the data set directory')
parser.add_argument('-pj', '--path-joblib', help='Path to joblib file from ivadomed containing the dataset splits.',
                    default=None, type=str)
parser.add_argument('-po', '--path-out', type=str, help='Path to the output directory where dataset json is saved')
parser.add_argument('-csuf', '--contrast-suffix', type=str, default='T1w', 
                    help='Contrast suffix used in the BIDS dataset')
args = parser.parse_args()


def main(args):

    root = args.path_data
    contrast = args.contrast_suffix

    # Get all subjects
    # the participants.tsv file might not be up-to-date, hence rely on the existing folders
    # subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
    # subjects = subjects_df['participant_id'].values.tolist()
    subjects = [subject for subject in os.listdir(root) if subject.startswith('sub-')]
    logger.info(f"Total number of subjects in the root directory: {len(subjects)}")
    

    if args.path_joblib is not None:
        # load information from the joblib to match train and test subjects
        joblib_file = os.path.join(args.path_joblib, 'split_datasets_all_seed=15.joblib')
        splits = joblib.load("split_datasets_all_seed=15.joblib")
        # get the subjects from the joblib file
        # train_subjects = sorted(list(set([sub.split('_')[0] for sub in splits['train']])))
        # val_subjects = sorted(list(set([sub.split('_')[0] for sub in splits['valid']])))
        test_subjects = sorted(list(set([sub.split('_')[0] for sub in splits['test']])))

    else:
        test_subjects = subjects

    logger.info(f"Number of testing subjects: {len(test_subjects)}")

    # keys to be defined in the dataset_0.json
    params = {}
    params["description"] = args.dataset_name
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
    params["reference"] = "University of Zurich"
    params["tensorImageSize"] = "3D"

    test_subjects_dict =  {"test": test_subjects}

    for name, subs_list in test_subjects_dict.items():

        temp_list = []
        for subject_no, subject in enumerate(subs_list):

            temp_data= {}

            temp_data["image"] = os.path.join(root, subject, 'anat', f"{subject}_{contrast}.nii.gz")
            if args.dataset_name == "sci-colorado":
                temp_data["label"] = os.path.join(root, "derivatives", "labels", subject, 'anat', f"{subject}_{contrast}_seg-manual.nii.gz")
            elif args.dataset_name == "basel-mp2rage-rpi":
                temp_data["label"] = os.path.join(root, "derivatives", "labels", subject, 'anat', f"{subject}_{contrast}_label-SC_seg.nii.gz")
            else:
                raise NotImplementedError(f"Dataset {args.dataset_name} not implemented yet.")
            
            if os.path.exists(temp_data["label"]) and os.path.exists(temp_data["image"]):
                temp_list.append(temp_data)
            else:
                logger.info(f"Subject {subject} does not have label or image file. Skipping it.")
        
        params[name] = temp_list
        logger.info(f"Number of images in {name} set: {len(temp_list)}")

    final_json = json.dumps(params, indent=4, sort_keys=True)
    jsonFile = open(args.path_out + "/" + f"{args.dataset_name}_dataset.json", "w")
    jsonFile.write(final_json)
    jsonFile.close()


if __name__ == "__main__":
    main(args)

    


