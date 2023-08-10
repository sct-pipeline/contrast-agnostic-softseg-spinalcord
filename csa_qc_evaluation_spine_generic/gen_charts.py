#!/usr/bin/env python
# -*- coding: utf-8
# Usage:
#    python gen_charts.py --contrasts T1w T2w T2star rec-average_dwi \
#        --predictions_folder ../duke/projects/ivadomed/contrast-agnostic-seg/csa_measures_pred/v1/ \
#        --baseline_folder ../duke/projects/ivadomed/contrast-agnostic-seg/archive_derivatives_softsegs-seg/contrast-agnostic-preprocess-all-2022-08-21-final/results

import os
import logging
import argparse
import sys
import pandas as pd
from scipy import stats
from charts_utils import macro_sd_violin, contrast_specific_pwd_violin, create_perf_df_sd, create_experiment_folder, create_perf_df_pwd, macro_sd_violin_preli
 
FNAME_LOG = 'log_stats.txt'

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrasts",
                        required=False,
                        nargs="*",
                        help="Contrasts to use for charts.",
                        default=['T1w', 'T2w', 'T2star', 'rec-average_dwi', 'flip-1_mt-on_MTS', 'flip-2_mt-off_MTS'])
    parser.add_argument("--gtypes",
                        required=False,
                        nargs="*",
                        default=["meanGT", 'hard'],
                        help="Ground truth types to consider for the charts. Possible values: 'hard' & 'meanGT'.")
    parser.add_argument("--augtypes",
                        required=False,
                        nargs="*",
                        default=["soft", 'hard'],
                        help="Augmentation types to consider for the charts. Possible values: 'hard' & 'soft'.")
    parser.add_argument("--baseline_folder",
                        required=True,
                        nargs=1,
                        help="Folder in which the CSA for the GT and meanGT are.")
    parser.add_argument("--predictions_folder",
                        required=True,
                        nargs=1,
                        help="Folder in which the CSA values for the segmentation" +
                        "predictions are (for the specified contrasts).")
    return parser

def extract_csv(csv_path: str) -> dict:
    """Extract csv values in the following format : {'patient': 'csa'}"""
    val_dic = {}
    dataset = pd.read_csv(csv_path)
    # Extract patient number in filename
    for idx, row in dataset.iterrows():
        patient = row.Filename.split('/')[-1].split('_')[0]
        val_dic[patient] = row['MEAN(area)']
        #print(f"Patient : {patient} | Val CSA : {val_dic[patient]}")

    return val_dic

def merge_csv(prediction_data_folder: str, 
              baseline_data_folder: str, 
              contrasts: list,
              gtypes: list,
              augtypes: list):
    """Creates dataframes containing CSA values and patients IDs for each
    found seed. It also creates a dataframe with CSA values for the preliminary
    experiment (meanGT vs manual). """
    csa_folders = os.listdir(prediction_data_folder)
    logger.info(f"\nCSA FOLDERS (Prediction) : {csa_folders}\n\n")
    main_dic = {}
    seeds_in_folders = []

    for fd in csa_folders:
        parts = fd.split('_')
        gtype = fd.split('_')[0]
        augtype = fd.split('_')[1]
        contrast = [s for s in contrasts if s in fd ]

        if len(contrast) == 0:
            contrast = 'all'
        else:
            contrast = contrast[0]

        seed = [part for part in parts if "seed" in part][0][5:]
        seeds_in_folders.append(int(seed))
        if contrast == 'all':
            csvs = [ff for ff in os.listdir(os.path.join(prediction_data_folder, fd, 'results')) if ff[-4:]=='.csv']
            for csv_path in csvs:
                absolute_csv_path = os.path.join(prediction_data_folder, fd, 'results', csv_path)
                if "dwi" in csv_path:
                    contrast = "rec-average_dwi"  # TODO: Refactor CSA folder names to only rec-average_dwi or dwi
                else:
                    contrast = [s for s in contrasts if s in csv_path][0]
                main_dic_key = "_".join([gtype, augtype, "all", contrast, f"seed={seed}"])
                val_dic = extract_csv(absolute_csv_path)
                logger.info(f"Folder : {fd} | Contrast : {contrast} | # patients:  {len(val_dic.keys())}")
                print(main_dic_key)
                main_dic[main_dic_key] = val_dic
        else:
            if contrast == "rec-average_dwi":
                csv_path = "csa_pred_" + "dwi" +'.csv'
            else: 
                csv_path = "csa_pred_" + contrast +'.csv'
            absolute_csv_path = os.path.join(prediction_data_folder, fd ,'results', csv_path)
            main_dic_key = "_".join([gtype, augtype, contrast, f"seed={seed}"])
            val_dic = extract_csv(absolute_csv_path)
            logger.info(f"Folder : {fd} | Contrast : {contrast} | # patients:  {len(val_dic.keys())}")
            main_dic[main_dic_key] = val_dic

    def check_participants_spec_all(participants_all: list, participants_spec: list, contrast: str, seed: int):
        """Verify that each patients (predictions) are the same for a specialist
        model and its associated generalist model."""
        participants_all = set(participants_all)
        participants_spec = set(participants_spec)
        seed_patients_right = participants_spec - participants_all.intersection(participants_spec)
        seed_patients_left = participants_all - participants_spec.intersection(participants_all)
        if seed_patients_right != set():
            logger.info(f"\nSeed {seed}| Missing {contrast} in spec model : {seed_patients_right}\n")
        if seed_patients_left != set():
            logger.info(f"\nSeed {seed}| Missing {contrast} in all model : {seed_patients_left}\n")
    seeds = list(set(seeds_in_folders))
    for s in seeds:
        for contrast in contrasts:
            for gtype in gtypes:
                for augtype in augtypes:
                    check_participants_spec_all(participants_all=list(main_dic[f"{gtype}_{augtype}_all_{contrast}_seed={s}"]),
                                                participants_spec=list(main_dic[f"{gtype}_{augtype}_{contrast}_seed={s}"]),
                                                contrast=contrast,
                                                seed=s)

    # Extract information from baseline CSVs
    baseline_csvs = [ff for ff in os.listdir(baseline_data_folder) if ff[-4:]=='.csv']
    for b_csv in baseline_csvs:
        gtype = b_csv[4:8]
        contrast = b_csv[:-4][9:]  # hard-coded because contrast can be 2 words
        #_, gtype, _, contrast = b_csv[:-4].split('_') 
        contrast = "GT_rec-average_dwi" if contrast == "GT_dwi" else contrast
        main_dic_key = "_".join(["manual", gtype, contrast])
        val_dic = extract_csv(os.path.join(baseline_data_folder, b_csv))

        logger.info(f"Baseline file : {'baseline/'+b_csv} | Contrast : {contrast} | # patients:  {len(val_dic.keys())}")
        main_dic[main_dic_key] = val_dic

    def get_common_patients(seeds: list, contrasts: list):
        """Get common patients for meanGT_soft models.
        TODO: Verify that all patients from other model types (GT type, Aug type)
        have the same patients as well. """
        seed2patients = {}

        for s in seeds:
            common_patients = None
            for contrast in contrasts:
                patients_spec = set(list(main_dic[f"meanGT_soft_{contrast}_seed={s}"]))
                patients_all = set(list(main_dic[f"meanGT_soft_all_{contrast}_seed={s}"]))
                if patients_spec != patients_all:
                    logging.warning(f"\n\nAll & spec patients are not matching for : Seed {s} - Contrast : {contrast} \n Patients spec : {patients_spec} \n Patients all : {patients_all}\n\n")
                if common_patients == None:
                    common_patients = patients_spec
                else:
                    common_patients = common_patients.intersection(patients_spec)
            seed2patients[str(s)] = common_patients
        return seed2patients

    logger.info(f"Main dic keys : {list(main_dic.keys())}")
    seed2patients = get_common_patients(seeds, contrasts)
    
    baseline_keys = [f"manual_{gtype}_GT_{c}" for gtype in ["hard", "soft"] for c in contrasts]
    model_keys_spec = [f"{gtype}_{augtype}_{contrast}" for gtype in gtypes for augtype in augtypes for contrast in contrasts]
    model_keys_all = [f"{gtype}_{augtype}_all_{contrast}" for gtype in gtypes for augtype in augtypes for contrast in contrasts]
    model_keys = model_keys_spec + model_keys_all

    final_results_dic_seed = []
    # One dataframe with CSA Values for each seed
    for k, s in enumerate(seeds): 
        patients = seed2patients[str(s)]
        final_dic = {kk: [] for kk in (["patient_id"]+baseline_keys+model_keys)}
        
        for p in patients:
            final_dic["patient_id"].append(p)
            for key_m in model_keys:
                final_dic[key_m].append(main_dic[(key_m+f"_seed={s}")][p])
            for key_b in baseline_keys:
                final_dic[key_b].append(main_dic[key_b][p])
        
        final_results_dic_seed.append(final_dic)

    # 1 dataframe for each seed
    dfs = {str(s): pd.DataFrame.from_dict(dic) for (s, dic) in zip(seeds, final_results_dic_seed)}
    
    ## Preliminary exp dataframe - meanGT VS hardGT 
    patients_inter = set(list(main_dic[f"manual_soft_GT_T1w"].keys()))
    manual_keys = [f"manual_{gtype}_GT_{c}" for gtype in ["hard", "soft"] for c in ["T1w", "T2w", "T2star", "rec-average_dwi", "flip-2_mt-off_MTS", "flip-1_mt-on_MTS"]]
    for manual_key in manual_keys:
        patients_inter = patients_inter.intersection(set(list(main_dic[manual_key].keys())))
    
    preli_dic = {kk: [] for kk in (["patient_id"]+manual_keys)}
    for p in patients_inter:
        preli_dic["patient_id"].append(p)
        for manual_key in manual_keys:
            preli_dic[manual_key].append(main_dic[manual_key][p])

    return dfs, pd.DataFrame.from_dict(preli_dic)


def compute_paired_t_test(x, y):

    return stats.ttest_rel(x, y, nan_policy='omit')


def main():

    exp_folder = create_experiment_folder()

    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(os.path.join(exp_folder, FNAME_LOG))
    logging.root.addHandler(fh)

    args = get_parser().parse_args()
    prediction_data_folder = args.predictions_folder[0]
    baseline_data_folder = args.baseline_folder[0]
    dfs, preli_df = merge_csv(prediction_data_folder, baseline_data_folder, 
                              contrasts=args.contrasts,
                              gtypes=args.gtypes,
                              augtypes=args.augtypes)

    # Manual - MeanGT VS HardGT effect
    methods_preli = ["manual_hard_GT", "manual_soft_GT"] # Benchmarks
    contrast_preli = args.contrasts
    perf_df_sd, macro_perf_sd_names = create_perf_df_sd(preli_df, methods_preli, contrast_preli)
    
    macro_sd_violin_preli(perf_df_sd, methods=macro_perf_sd_names, outfile=os.path.join(exp_folder, f"preli_meanGT_VS_default.png"))
    logger.info(f"Length preliminary dataset:  {len(preli_df)}") # 103

    # Individual seeds
    methods = ["manual_hard_GT", "manual_soft_GT", "hard_hard", "hard_soft",
               "meanGT_soft", "meanGT_soft_all"]

    contrasts = [c for c in args.contrasts if c != "T2w"]  #["T1w", "T2star"]
    ref_contrast = "T2w"
    if ref_contrast not in args.contrasts:
        raise ValueError("T2w is the default reference contrast for PWD charts. Please make sure you include this contrast in the arguments.")

    # Macro performance plots 
    for (s, dataframe) in dfs.items():
        perf_df_pwd, macro_perf_pwd_names = create_perf_df_pwd(dataframe, methods, contrasts, ref_contrast)
        contrast_specific_pwd_violin(perf_df_pwd, methods, contrasts, outfile=os.path.join(exp_folder, f"contrast_specific_PWD_seed{s}.png"))
        perf_df_sd, macro_perf_sd_names = create_perf_df_sd(dataframe, methods, contrasts+[ref_contrast])
        #macro_sd_violin(perf_df_sd, methods=macro_perf_sd_names, outfile=os.path.join(exp_folder, f"macroSD_seed_{s}.png"))

    # Seed Aggregation - Macro performance plots across seeds
    agg_df = pd.concat(dfs.values())
    duplicates = agg_df.duplicated(subset="patient_id")
    logger.info(f"# Duplicates : {sum(duplicates)}/{len(duplicates)} \n Duplicate patients :\n {agg_df['patient_id'][duplicates]}")
    perf_df_sd, macro_perf_sd_names = create_perf_df_sd(agg_df, methods, contrasts+[ref_contrast])
    print(macro_perf_sd_names)
    macro_perf_sd_names.remove('hard_soft_perf_sd')
    macro_sd_violin(perf_df_sd, methods=macro_perf_sd_names, outfile=os.path.join(exp_folder, f"macroSD_allseeds.png"))
    
    # Compute paired T-test
    # Compare soft augmentation vs soft avg
    results_2_vs_3 = compute_paired_t_test(perf_df_sd['hard_soft_perf_sd'], perf_df_sd['meanGT_soft_perf_sd'])
    logger.info("Paired T-test: soft augmentation vs soft average: {}".format(results_2_vs_3))
    # Compare per contrast vs all contrast
    results_3_vs_4 = compute_paired_t_test(perf_df_sd['meanGT_soft_perf_sd'], perf_df_sd['meanGT_soft_all_perf_sd'])
    logger.info("Paired T-test: per contrast vs all contrast: {}".format(results_3_vs_4))
    # Compare meant GT vs all contrast
    results_GT_vs_4 = compute_paired_t_test(perf_df_sd['manual_hard_GT_perf_sd'], perf_df_sd['meanGT_soft_all_perf_sd'])
    logger.info("Paired T-test: meant GT vs all contrast: {}".format(results_GT_vs_4))

    #['manual_hard_GT_perf_sd', 'manual_soft_GT_perf_sd', 'hard_hard_perf_sd', 'hard_soft_perf_sd', 'meanGT_soft_perf_sd', 'meanGT_soft_all_perf_sd']

if __name__ == "__main__":
    main()
