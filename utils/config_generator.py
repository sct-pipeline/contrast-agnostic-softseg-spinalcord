"""
Script that takes as input an ivadomed config (JSON) & replaces the following dynamic string params:
    - <CONTRAST>: The contrast name, you can name it however you like, e.g., T1w, T2w, T1wANDT2w
    - <CONTRAST_PARAMS>: The contrast param name, has to match file suffix unlike <CONTRAST>
    - <FNAME_SPLIT>: The .joblib file to use to split the dataset into train-val-test
    - <SEED>: The seed to use
This script creates several config files for different contrast and seed scenarios
Inputs:
    --config: Path to the base ivadomed config (JSON) with dynamic string params
    --datasets: List of BIDS dataset folders
    --ofolder: Output folder for .joblib files
    --contrasts: Replaces <CONTRAST>
    --seeds: Replaces <SEED>
Example usage:
python config_generator.py --config config/meanGT_soft.json \
                           --datasets /home/GRAMES.POLYMTL.CA/uzmac/duke/projects/ivadomed/contrast-agnostic-seg/contrast-agnostic-preprocess-all-2022-08-21-final/data_processed_clean \
                           --ofolder joblibs \
                           --contrasts T1w T2w T2star rec-average_dwi \
                           --seeds 42 15 34 98 62
"""

import glob
import os
import json
import pandas as pd
import joblib
import random
import numpy as np
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the ivadomed config (JSON)")
parser.add_argument("--datasets", required=True, nargs="*", help="List of BIDS dataset folders")
parser.add_argument("--ofolder", type=str, default="joblibs", help="Output folder for .joblib files")
parser.add_argument("--contrasts", required=True, nargs="*", help="Contrasts to generate .joblib files for and replace <CONTRAST>")
parser.add_argument("--seeds", type=int, default=42, nargs="*", help="Seed for randomly distributing subjects into train, validation, and testing and replaces <SEED>")
args = parser.parse_args()

# Check validity of user inputs
assert args.config.endswith('.json')
assert all([os.path.exists(dataset) for dataset in args.datasets])

# Create output directory if it doesn't exists
if not os.path.exists(args.ofolder):
    os.makedirs(args.ofolder)

for seed in args.seeds:
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Merge multiple TSV files from multiple BIDS datasets (if applicable) into single data frame
    df_merged = pd.read_table(os.path.join(args.datasets[0], 'participants.tsv'), encoding="ISO-8859-1")
    # Convert to string to get rid of potential TypeError during merging within the same column
    df_merged = df_merged.astype(str)
    # Add the BIDS path to the data frame
    df_merged['bids_path'] = [args.datasets[0]] * len(df_merged)

    for i in range(1, len(args.datasets)):
        df_next = pd.read_table(os.path.join(args.datasets[i], 'participants.tsv'), encoding="ISO-8859-1")
        df_next = df_next.astype(str)
        df_next['bids_path'] = [args.datasets[i]] * len(df_next)
        # Merge the .tsv files (This keeps also non-overlapping fields)
        df_merged = pd.merge(left=df_merged, right=df_next, how='outer')

    print('Got following BIDS datasets: ', set(df_merged['bids_path'].values.tolist()))
    print('Generating joblibs for following contrasts: ', args.contrasts)

    # Set train, validation, and test split percentages
    pct_train, pct_validation, pct_test = 0.6, 0.2, 0.2
    assert pct_train + pct_validation + pct_test == 1.0

    # Split subjects into train, validation, and test groups
    subs = df_merged['participant_id'].values.tolist()
    np.random.shuffle(subs)
    train_subs = subs[:int(len(subs) * pct_train)]
    validation_subs = subs[int(len(subs) * pct_train): int(len(subs) * (pct_train + pct_validation))]
    test_subs = subs[int(len(subs) * (pct_train + pct_validation)):]
    print('Got %d train, %d validation, and %d test subjects!' % (len(train_subs), len(validation_subs), len(test_subs)))

    jobdicts = []
    for contrast in args.contrasts:
        train_subs_cur, validation_subs_cur, test_subs_cur = [], [], []
        for sub in train_subs:
            bids_path = df_merged[df_merged['participant_id'] == sub]['bids_path'].values.tolist()[0]
            files = glob.glob(os.path.join(bids_path, sub) + '/**/*%s.nii.gz' % contrast, recursive=True)
            if len(files) != 0:
                train_subs_cur.append('%s_%s.nii.gz' % (sub, contrast))
        for sub in validation_subs:
            bids_path = df_merged[df_merged['participant_id'] == sub]['bids_path'].values.tolist()[0]
            files = glob.glob(os.path.join(bids_path, sub) + '/**/*%s.nii.gz' % contrast, recursive=True)
            if len(files) != 0:
                validation_subs_cur.append('%s_%s.nii.gz' % (sub, contrast))
        for sub in test_subs:
            bids_path = df_merged[df_merged['participant_id'] == sub]['bids_path'].values.tolist()[0]
            files = glob.glob(os.path.join(bids_path, sub) + '/**/*%s.nii.gz' % contrast, recursive=True)
            if len(files) != 0:
                test_subs_cur.append('%s_%s.nii.gz' % (sub, contrast))

        jobdict = {"train": train_subs_cur, "valid": validation_subs_cur, "test": test_subs_cur}
        jobdicts.append(jobdict)
        joblib.dump(jobdict, os.path.join(args.ofolder, "split_datasets_%s_seed=%s.joblib" % (contrast, seed)))

    # Generate one final joblib for all contrast training
    jobdict_all = {"train": [], "valid": [], "test": []}
    for i in range(len(jobdicts)):
        jobdict = jobdicts[i]
        for key, values in jobdict.items():
            for value in values:
                jobdict_all[key].append(value)
    joblib.dump(jobdict_all, os.path.join(args.ofolder, "split_datasets_all_seed=%s.joblib" % seed))

# Read config
config_dir, config_fname = os.path.dirname(args.config), os.path.basename(args.config)
config = json.load(open(args.config))

for seed in args.seeds:
    seed = str(seed)
    contrast_params = []
    for contrast in args.contrasts:
        # Adjust contrast param for specific use cases
        contrast_param = contrast
        if contrast == 'rec-average_dwi':
            contrast_param = 'dwi'
        contrast_params.append(contrast_param)

        # Convert config to string and replace the supplied parameters
        config_str = json.dumps(config).\
            replace('<CONTRAST>', contrast).\
            replace('<CONTRAST_PARAMS>', contrast_param).\
            replace('<FNAME_SPLIT>', os.path.join(args.ofolder, 'split_datasets_%s_seed=%s.joblib' % (contrast, seed))).\
            replace('<SEED>', seed)

        # Get config back with replacements
        config_filled = json.loads(config_str)
        config_filled_path = os.path.join(
            config_dir,
            '%s_%s_seed=%s.json' % (os.path.splitext(config_fname)[0], contrast, seed)
        )

        # Write back the config
        with open(config_filled_path, 'w', encoding='utf-8') as f:
            json.dump(config_filled, f, ensure_ascii=False, indent=4)
        print('Created config: %s' % config_filled_path)

    # Convert config to string and replace the supplied parameters
    config_str = json.dumps(config). \
        replace('<CONTRAST>', 'all'). \
        replace('<CONTRAST_PARAMS>', ','.join(contrast_params)). \
        replace('<FNAME_SPLIT>', os.path.join(args.ofolder, 'split_datasets_all_seed=%s.joblib' % seed)). \
        replace('<SEED>', seed)

    # Get config back with replacements
    config_filled = json.loads(config_str)

    # Write combined contrasts in a list format
    for key in ["training_validation", "testing"]:
        config_filled["loader_parameters"]["contrast_params"][key] = config_filled["loader_parameters"]["contrast_params"][key][0].split(',')

    config_filled_path = os.path.join(
        config_dir,
        '%s_%s_seed=%s.json' % (os.path.splitext(config_fname)[0], 'all', seed)
    )

    # Write back the config
    with open(config_filled_path, 'w', encoding='utf-8') as f:
        json.dump(config_filled, f, ensure_ascii=False, indent=4)
    print('Created config: %s' % config_filled_path)