import pandas as pd
import numpy as np
import scipy.stats
import os 
import json

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h #m-h, m+h

def mean_std(data):
    a = 1.0 * np.array(data)
    return np.mean(a), np.std(a)


def compute_metrics(data_paths):
    """Stores DICE and Relative Difference Volume (RVD) of CSV segmentations 
    results in list_paths for each model."""
    results = {}
    for dp in data_paths:
        data_path = f"./spine-generic-test-results/{dp}/evaluation_3Dmetrics.csv"
        df = pd.read_csv(data_path)
        volume_diff = df["RVD"]
        
        avg_vol, conf_vol = mean_std(volume_diff) 
        avg_dice, conf_dice = mean_std(df["Dice"])
        print(f"Model {dp} | \n \t DICE : {avg_dice*100} ± {conf_dice*100} \n \t VOLUME : {avg_vol*100} ± {conf_vol*100}")
        results[dp] = {"avg_vol": avg_vol * 100, "conf_vol": conf_vol * 100,"avg_dice": avg_dice * 100, "conf_dice": conf_dice * 100}

    return results

def get_metric_values(list_paths):
    """Accumulates DICE and Relative Difference Volume (RVD) of CSV segmentations 
    results in list_paths."""
    dices = []
    volume_diffs = []
    for dp in list_paths:
        data_path = f"./spine-generic-test-results/{dp}/evaluation_3Dmetrics.csv"
        df = pd.read_csv(data_path)
        volume_diff = df["RVD"]
        dice = df["Dice"]
        dices = np.concatenate((dices, np.array(dice)), axis=None)
        volume_diffs = np.concatenate((volume_diffs, np.array(volume_diff)), axis=None)
    return *mean_std(dices), *mean_std(volume_diffs)

    
if __name__ == "__main__":
    
    contrasts = ["T1w", "T2w", "T2star", "rec-average_dwi", "flip-1_mt-on_MTS", "flip-2_mt-off_MTS", "all"]
    spec_contrasts = [c for c in contrasts if c != "all"]
    methods = ["meanGT", "hard"]
    aug_methods = ["hard", "soft"]
    results_path = "./temp_results"
    list_paths = [f"{method}_{aug_method}_{contrast}_seed=15" for method in methods for aug_method in aug_methods for contrast in contrasts]
    
    ## Aggregated results (per_contrast and all_contrast)
    aggregated_results = {}
    for method in methods:
        for aug_method in aug_methods:
            for tp in ["percontrast", "allcontrast"]:
                if tp == "percontrast":
                    avg_dice, conf_dice, avg_vol, conf_vol = get_metric_values([f"{method}_{aug_method}_{c}_seed=15" for c in spec_contrasts])
                    
                else:
                    avg_dice, conf_dice, avg_vol, conf_vol = get_metric_values([f"{method}_{aug_method}_all_seed=15"])
                aggregated_results[f"{method}_{aug_method}_{tp}"] = {"avg_vol": avg_vol * 100, "conf_vol": conf_vol * 100,"avg_dice": avg_dice * 100, "conf_dice": conf_dice * 100}
                    
    with open("./spine-generic-test-results/miccai_aggregated_results.json", "w") as fp:
        json.dump(aggregated_results, fp, indent=4)
    with open("./spine-generic-test-results/miccai_aggregated_table_export.txt", "w") as fp:
        for key, value in aggregated_results.items():
            line = "\\texttt" + "{" + f"{key}" + "}" + f"&  {round(value['avg_dice'],2)} ± {round(value['conf_dice'],2)} & {round(value['avg_vol'],2)} ± {round(value['conf_vol'],2)} \\\\ \n"
            fp.write(line)

    ## All 28 models results
    key_paths = {}
    for method in methods:
        c_m = "soft\\_average" if method == "meanGT" else "hard"
        for aug_method in aug_methods:
            #ag_m = "soft"
            for contrast in contrasts:
                if "flip-1" in contrast :
                    c_con = "MTS-ON"
                elif "flip-2" in contrast : 
                    c_con = "MTS-OFF"
                elif "all" in contrast:
                    c_con = "allcontrast"
                elif "T2star" in contrast:
                    c_con = "T2star"
                elif "dwi" in contrast:
                    c_con = "DWI"
                else:
                    c_con = contrast
                key_paths[f"{method}_{aug_method}_{contrast}_seed=15"] = ("DL\_" + f"{c_m}" + "\_" + f"{aug_method}" + "\_" + f"{c_con}" )

    results = compute_metrics(list(key_paths.keys()))
    with open ("./spine-generic-test-results/miccai_results.json", "w") as fp:
        json.dump(results, fp, indent=4)
    
    with open("./spine-generic-test-results/miccai_table_export.txt", "w") as fp:
        names = []
        for key, value in results.items():
            names.append(key + "\n")
            line = "\\texttt" + "{" + f"{key_paths[key]}" + "}" + f"&  {round(value['avg_dice'],2)} ± {round(value['conf_dice'],2)} & {round(value['avg_vol'],2)} ± {round(value['conf_vol'],2)} \\\\ \n"  
            fp.write(line)
    