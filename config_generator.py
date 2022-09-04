"""
Script that generates .joblib files for all contrasts in a BIDS dataset such that all train/test
splits for each contrast setting use the same subjects throughout.
Adapted from https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/blob/main/create_training_joblib.py

Inputs:
    --bids_datasets_list: List of BIDS dataset folders
    --ofolder: Output folder for .joblib files
    --contrasts: Contrasts to generate .joblib files for
    --seed: Seed for randomly distributing subjects into train, validation, and testing

Example usage:
python joblib_generator.py --datasets /home/GRAMES.POLYMTL.CA/uzmac/duke/projects/ivadomed/contrast-agnostic-seg/contrast-agnostic-preprocess-all-2022-08-19/data_processed_clean --ofolder joblibs --contrasts T1w T2w T2star rec-average_dwi --seed 42
"""

import glob
import os
import pandas as pd
import joblib
import random
import numpy as np
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", required=True, nargs="*", help="List of BIDS dataset folders")
parser.add_argument("--ofolder", type=str, default="joblibs", help="Output folder for .joblib files")
parser.add_argument("--contrasts", required=True, nargs="*", help="Contrasts to generate .joblib files for")
parser.add_argument("--seed", type=int, default=42, help="Seed for randomly distributing subjects into train, validation, and testing")
args = parser.parse_args()

# Create output directory if it doesn't exists
if not os.path.exists(args.ofolder):
    os.makedirs(args.ofolder)

# Set random seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)

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
    joblib.dump(jobdict, os.path.join(args.ofolder, "split_datasets_%s_seed=%s.joblib" % (contrast, args.seed)))

# Generate one final joblib for all contrast training
jobdict_all = defaultdict(list)
for i in range(len(jobdicts)):
    jobdict = jobdicts[i]
    for key, value in jobdict.items():
        jobdict_all[key].append(value)
joblib.dump(jobdict_all, os.path.join(args.ofolder, "split_datasets_all_seed=%s.joblib" % args.seed))
