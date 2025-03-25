#!/bin/bash
# Wrapper to the .py script containing code to plot the morphometrics on spine generic test set


# path to the input folder containing the CSV files for each model release
PATH_IN=$1

# Run script to merge 
python csa_generate_figures/analyse_csa_across_releases.py -i $PATH_IN

echo "=============================="
echo "Generated CSA plots! "
echo "=============================="
