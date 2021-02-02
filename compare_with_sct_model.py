# This script gathers the derivatives from the files/subjects that were used in the testing set
# Then the sct_deepseg will be used on each folder to create an evaluation_3Dmetrics.csv file within each folder for
# comparison with the newly created models

# In steps, what this function does:
# 1. The files used in the newly trained model are identified
# 2. SCT_DEEPSEG_SC performs segmentations on them - all segmentations are saved within outputFolder/sct_deepseg
# 3. Dice scores are computed between the derivatives of the original files, and the SCT segmentations
# 4. The scores are gathered in outputFolder/ basename(logfolder+"_SCT")/results_eval/evaluation_3Dmetrics.csv

# Creating a similar evaluation_3Dmetrics.csv allows usage of the violinplots function we already have for comparison

# Inputs:
# --logfolder: Log folder of newly trained mode. The SCT segmentation will be compared to that
# --ofolder: This is the parent folder where all new files will be created : a new subfolder will be
#            created within it for each log folder with the folderName of the logFolder and the suffix _SCT")

# Example usage:
# python3 compare_with_sct_model --logfolder newlyTrainedModel --ofolder outputFolder

# Konstantinos Nasiotis 2021


import os
import pandas as pd
from shutil import copyfile
import subprocess
import argparse
import json


def compare_to_sct(log_folder="/home/nas/PycharmProjects/ivadomed-personal-scripts/ResultsNewModel/Artificial_Log_folders/t1w_new_model",
                   output_Folder_to_create_SCT_log_folders_in="/home/nas/PycharmProjects/ivadomed-personal-scripts/ResultsNewModel"):

    # Gather used parameters from the training
    config_file = os.path.join(log_folder, 'config_file.json')
    with open(config_file) as json_file:
        parameters = json.load(json_file)

    if isinstance(parameters['loader_parameters']['bids_path'], str):
        BIDS_path = parameters['loader_parameters']['bids_path']
    elif isinstance(parameters['loader_parameters']['bids_path'], list):
        print("THIS NEEDS TO GENERALIZE TO THE NEW LOADER WITH MULTIPLE BIDS FOLDERS")
        BIDS_path = parameters['loader_parameters']['bids_path'][0]  # Generalize for multiple
    suffix = parameters['loader_parameters']['target_suffix'][0]  # Only "_seg-manual" is expected here - maybe generalize

    # Get the scores that the new model achieved - This is what will be used for comparison to SCT performance on the
    # same files
    results_file = os.path.join(log_folder, "results_eval", "evaluation_3Dmetrics.csv")
    results = pd.read_csv(results_file)

    # Get the subjectID in a list
    subjects = [x.replace("_T1w", "").replace("_T2w", "").replace("_T1star", "") for x in results["image_id"].to_list()]
    subjects_with_modality_string = results["image_id"].to_list()

    # CREATE NECESSARY FOLDERS
    if not os.path.exists(os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder)+"_SCT")):
        os.mkdir(os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder)+"_SCT"))
        os.mkdir(os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder) + "_SCT", "derivatives"))

    # Collect the absolute path of the files that were used for testing
    files_to_run_sct_deepseg_on = []  # This will hold all the original files
    gt_to_run_dice_score = []  # This will hold the derivatives
    all_contrasts = []  # This will hold the modality of each file used

    copy_files = 1  #FOR DEBUGGING - copies the files that were used in training (and their derivatives) to the output folder

    for single_subject_with_modality_string in subjects_with_modality_string:
        filename = single_subject_with_modality_string + '.nii.gz'
        subject_WITHOUT_modality_string = single_subject_with_modality_string.replace("_T1w", "").replace("_T2w", "").replace("_T1star", "")
        modality = single_subject_with_modality_string.split("_")[-1]
        all_contrasts.append(modality)
        if os.path.exists(os.path.join(BIDS_path, subject_WITHOUT_modality_string, 'anat', filename)):
            if copy_files:
                copyfile(os.path.join(BIDS_path, subject_WITHOUT_modality_string, 'anat', filename),
                         os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder) + "_SCT", filename))
            files_to_run_sct_deepseg_on.append(os.path.join(BIDS_path, subject_WITHOUT_modality_string, 'anat', filename))

        # Copy the derivatives
        derivative_filename = single_subject_with_modality_string + suffix + '.nii.gz'
        if os.path.exists(os.path.join(BIDS_path, 'derivatives', 'labels', subject_WITHOUT_modality_string, 'anat', derivative_filename)):
            if copy_files:
                copyfile(os.path.join(BIDS_path, 'derivatives', 'labels', subject_WITHOUT_modality_string, 'anat', derivative_filename),
                         os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder) + "_SCT", "derivatives", derivative_filename))

            gt_to_run_dice_score.append(os.path.join(BIDS_path, 'derivatives', 'labels', subject_WITHOUT_modality_string, 'anat', derivative_filename))

    # Now run sct_deep_seg_sc on each file - already computed segmentations will be skipped
    # This creates a pool of all possible files within the testing set that was used in the log_folder

    sct_deepseg_folder = os.path.join(output_Folder_to_create_SCT_log_folders_in, 'sct_deepseg')

    if not os.path.exists(sct_deepseg_folder):
        os.mkdir(sct_deepseg_folder)

    sct_segmented_files_to_run_dice_scores_on = []

    run_segmentation = 0
    if run_segmentation:
        for i in range(len(files_to_run_sct_deepseg_on)):

            FileFullPath = files_to_run_sct_deepseg_on[i]
            filename = os.path.basename(FileFullPath)
            filename = filename.replace(".nii.gz", "") + '_seg-sct.nii.gz'

            # Get appropriate input for SCT contrast
            contrast = all_contrasts[i]
            if contrast == "T1w":
                contrast_sct_input = "t1"
            elif contrast == "T2w":
                contrast_sct_input = "t2"
            elif contrast == "T2star":
                contrast_sct_input = "t2s"

            # Do the segmentation if not already done it before - Consider improving and using batch sct
            if not os.path.exists(os.path.join(sct_deepseg_folder, filename)):
                os.system('/home/nas/PycharmProjects/spinalcordtoolbox/bin/sct_deepseg_sc -i ' + FileFullPath
                          + " -c " + contrast_sct_input + " -o " + os.path.join(sct_deepseg_folder, filename))
                #subprocess.run(["sct_deepseg_sc", "-i", FileFullPath,
                #          "-c", contrast_sct_input, "-o", os.path.join(sct_deepseg_folder, filename)])

            else:
                print('Already segmented: ' + os.path.join(sct_deepseg_folder, filename))
            sct_segmented_files_to_run_dice_scores_on.append(os.path.join(sct_deepseg_folder, filename))

    # Now compute the dice_scores between the sct-segmentation and the ground-truth
    # and collect the values in a .csv file
    subject_labels = []
    diceScores = []

    for File in gt_to_run_dice_score:
        basename = os.path.basename(File.replace(suffix + ".nii.gz", ""))
        sct_file_fullpath = os.path.join(output_Folder_to_create_SCT_log_folders_in, "sct_deepseg", basename + "_seg-sct.nii.gz")

        if os.path.exists(sct_file_fullpath):
            diceScoreFile = os.path.join(output_Folder_to_create_SCT_log_folders_in, "dice_score.txt")  # Temp file

            os.system("/home/nas/PycharmProjects/spinalcordtoolbox/bin/sct_dice_coefficient -i " + File + " -d " +
                      sct_file_fullpath + " -o " + diceScoreFile)

            with open(diceScoreFile) as f:
                text = f.read()
                try:
                    diceScore = float(text.replace('3D Dice coefficient = ', ''))
                except:
                    diceScore = float("nan")

            # Append results
            subject_labels.append(basename)
            diceScores.append(diceScore)

    # Remove temp file
    os.remove(diceScoreFile)

    # Export all results to a .csv file within the "logFolder_SCT"
    df = pd.DataFrame({'image_id': subject_labels,
                       'dice_class0': diceScores})

    if not os.path.exists(os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder)+"_SCT", "results_eval")):
        os.mkdir(os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder)+"_SCT", "results_eval"))

    df.to_csv(os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder)+"_SCT", "results_eval", 'evaluation_3Dmetrics.csv'))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfolder", required=True, nargs="*", dest="logfolder",
                        help="Log folder of newly trained mode. The SCT segmentation will be compared to that")
    parser.add_argument("--ofolder", required=True, nargs="*", dest="outputFolder",
                        help="This is the parent folder where all new files will be created - a new subfolder will be" +
                             "created within it for each log folder")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Run comparison
    compare_to_sct(args.logfolder[0], args.outputFolder[0])


if __name__ == '__main__':
    main()
