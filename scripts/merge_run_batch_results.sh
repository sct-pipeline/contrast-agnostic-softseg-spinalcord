#!/bin/bash

# ==============================
# DEFINE GLOBAL VARIABLES
# ==============================

# path to the input folder containing the individual batch processing results
PATH_IN=$1
# path to the output folder where the merged csv will be saved
PATH_OUT=$2

# Run script to merge 
python scripts/merge_csvs.py -path-results $PATH_IN -path-output $PATH_OUT