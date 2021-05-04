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
# --copyfiles: FOR DEBUGGING - copies the files that were used in training (and their derivatives) to the output folder,
#              default: False

# Example usage:
# python3 compare_with_sct_model --logfolders newlyTrainedModel1 newlyTrainedModel2 --ofolder outputFolder --copyfiles False

# Konstantinos Nasiotis 2021


import os
import pandas as pd
from shutil import copyfile
import subprocess
import argparse
import json
import random
import platform
import multiprocessing as mp


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfolders", required=True, nargs="*", dest="logfolders",
                        help="Log folder of newly trained models. The SCT segmentation will be compared to that")
    parser.add_argument("--ofolder", required=True, nargs=1, dest="outputFolder",
                        help="This is the parent folder where all new files will be created - a new subfolder will be" +
                             "created within it for each log folder")
    parser.add_argument("--copyfiles", required=False, nargs=1, dest="copyfiles", default=False,
                        help="This is the parent folder where all new files will be created - a new subfolder will be" +
                             "created within it for each log folder")
    return parser


def compare_to_sct(log_folder,
                   output_Folder_to_create_SCT_log_folders_in,
                   copy_files=False):

    # Path for spinalcordtoolbox
    node = platform.node()
    if "acheron" in node:
        SCT_PATH = "/home/nas/PycharmProjects/spinalcordtoolbox/bin/"
    elif "rosenberg" in node:
        SCT_PATH = "/home/GRAMES.POLYMTL.CA/u111358/sct_5.1.0/bin/"
    else:
        raise NameError("need to specify a path for the sct toolbox")

    # Gather used parameters from the training
    config_file = os.path.join(log_folder, 'config_file.json')
    with open(config_file) as json_file:
        parameters = json.load(json_file)

    if isinstance(parameters['loader_parameters']['path_data'], str):
        BIDS_path = [parameters['loader_parameters']['path_data']]  # Convert to list
    elif isinstance(parameters['loader_parameters']['path_data'], list):
        BIDS_path = parameters['loader_parameters']['path_data']  # Generalize for multiple
    suffix = parameters['loader_parameters']['target_suffix'][0]  # Only "_seg-manual" is expected here - maybe generalize

    # Get the scores that the new model achieved - This is what will be used for comparison to SCT performance on the
    # same files
    results_file = os.path.join(log_folder, "results_eval", "evaluation_3Dmetrics.csv")
    results = pd.read_csv(results_file)

    # Get the subjectID in a list
    subjects_with_modality_string = results["image_id"].to_list()

    # CREATE NECESSARY FOLDERS
    if not os.path.exists(os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder)+"_SCT")):
        os.mkdir(os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder)+"_SCT"))
        os.mkdir(os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder) + "_SCT", "derivatives"))

    # Collect the absolute path of the files that were used for testing
    files_to_run_sct_deepseg_on = []  # This will hold all the original files
    gt_to_run_dice_score = []  # This will hold the derivatives

    # Randomization helps in parallel processing when running this code in multiple instances when segmenting
    random.shuffle(subjects_with_modality_string)

    # Creates the centerline for the
    create_centerline = 1

    for single_subject_with_modality_string in subjects_with_modality_string:
        filename = single_subject_with_modality_string + '.nii.gz'
        subject_WITHOUT_modality_string = single_subject_with_modality_string.replace("_T1w", "").replace("_T2w", "").replace("_T2star", "")

        for single_bids_folder in BIDS_path:
            file = os.path.join(single_bids_folder, subject_WITHOUT_modality_string, 'anat', filename)
            if os.path.exists(file):
                if copy_files:
                    copyfile(file,
                             os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder) + "_SCT", filename))

                if create_centerline:
                    centerline_file = os.path.join(single_bids_folder, 'derivatives', 'labels',
                                                   subject_WITHOUT_modality_string, 'anat',
                                                   single_subject_with_modality_string+"_centerline") # Don't add .nii.gz here - sct bug

                    # Get appropriate input for SCT contrast
                    contrast = file.split("_")[-1].replace(".nii.gz", "")
                    if contrast == "T1w":
                        contrast_sct_input = "t1"
                    elif contrast == "T2w":
                        contrast_sct_input = "t2"
                    elif contrast == "T2star":
                        contrast_sct_input = "t2s"

                    if not os.path.exists(centerline_file+".nii.gz"):
                        os.system(os.path.join(SCT_PATH, "sct_get_centerline") +
                                  " -i " + file +
                                  " -c " + contrast_sct_input +
                                  " -o " + centerline_file)

                files_to_run_sct_deepseg_on.append(os.path.join(single_bids_folder, subject_WITHOUT_modality_string, 'anat', filename))

            # Copy the derivatives
            derivative_filename = single_subject_with_modality_string + suffix + '.nii.gz'
            if os.path.exists(os.path.join(single_bids_folder, 'derivatives', 'labels', subject_WITHOUT_modality_string, 'anat', derivative_filename)):
                if copy_files:
                    copyfile(os.path.join(single_bids_folder, 'derivatives', 'labels', subject_WITHOUT_modality_string, 'anat', derivative_filename),
                             os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder) + "_SCT", "derivatives", derivative_filename))

                gt_to_run_dice_score.append(os.path.join(single_bids_folder, 'derivatives', 'labels', subject_WITHOUT_modality_string, 'anat', derivative_filename))

    # Now run sct_deep_seg_sc on each file - already computed segmentations will be skipped
    # This creates a pool of all possible files within the testing set that was used in the log_folder

    sct_deepseg_folder = os.path.join(output_Folder_to_create_SCT_log_folders_in, 'sct_deepseg')

    if not os.path.exists(sct_deepseg_folder):
        os.mkdir(sct_deepseg_folder)

    sct_segmented_files_to_run_dice_scores_on = []

    # Randomization helps in parallel processing
    random.shuffle(files_to_run_sct_deepseg_on)

    run_segmentation = 1
    if run_segmentation:
        for FileFullPath in files_to_run_sct_deepseg_on:  # Randomization helps in parallel processing

            filename = os.path.basename(FileFullPath)
            filename = filename.replace(".nii.gz", "_seg-sct.nii.gz")

            # Get appropriate input for SCT contrast
            contrast = FileFullPath.split("_")[-1].replace(".nii.gz", "")
            if contrast == "T1w":
                contrast_sct_input = "t1"
            elif contrast == "T2w":
                contrast_sct_input = "t2"
            elif contrast == "T2star":
                contrast_sct_input = "t2s"

            # Do the segmentation if not already done it before - Consider improving and using batch sct
            if not os.path.exists(os.path.join(sct_deepseg_folder, filename)):
                out = os.system(os.path.join(SCT_PATH, "sct_deepseg_sc") +
                          " -i " + FileFullPath +
                          " -c " + contrast_sct_input +
                          " -o " + os.path.join(sct_deepseg_folder, filename))
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
            diceScoreFile = os.path.join(output_Folder_to_create_SCT_log_folders_in,
                                         os.path.basename(log_folder) + "_SCT", "dice_score.txt")  # Temp file

            os.system(os.path.join(SCT_PATH, "sct_dice_coefficient") +
                      " -i " + File +
                      " -d " + sct_file_fullpath +
                      " -o " + diceScoreFile)

            with open(diceScoreFile) as f:
                text = f.read()
                try:
                    diceScore = float(text.replace('3D Dice coefficient = ', ''))
                except:
                    diceScore = float("nan")

            # Append results
            subject_labels.append(basename)
            diceScores.append(diceScore)

    # Cleanup
    os.remove(diceScoreFile)

    # Export all results to a .csv file within the "logFolder_SCT"
    df = pd.DataFrame({'image_id': subject_labels,
                       'dice_class0': diceScores})

    if not os.path.exists(os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder)+"_SCT", "results_eval")):
        os.mkdir(os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder)+"_SCT", "results_eval"))

    df.to_csv(os.path.join(output_Folder_to_create_SCT_log_folders_in, os.path.basename(log_folder)+"_SCT", "results_eval", 'evaluation_3Dmetrics.csv'))


def main():
    parser = get_parser()
    args = parser.parse_args()

    run_parallel = True
    if run_parallel:
        # Parallelize processing
        print('Starting parallel processing')
        #pool = mp.Pool(mp.cpu_count() - 2)
        pool = mp.Pool(len(args.logfolders))  # This should be ok
        results = [pool.apply_async(compare_to_sct, args=(logfolder, args.outputFolder[0],args.copyfiles)) for logfolder in args.logfolders]
        pool.close()
        pool.join()
        print('Just finished parallel processing')
    else:
        for logfolder in args.logfolders:
            compare_to_sct(log_folders=logfolder,
                           output_Folder_to_create_SCT_log_folders_in=args.outputFolder[0],
                           copy_files=args.copyfiles)


if __name__ == '__main__':
    main()
