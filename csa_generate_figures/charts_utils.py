from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

#plt.style.use('seaborn')
#plt.style.use('seaborn-ticks')



def create_perf_df_pwd(dataframe, methods, contrasts, ref_contrast="t2w", perf_suffix="_perf_pwd"):
    """Creates a copy of the original dataframe containing 2
    transformations:
        1) Transforms raw CSA into CSA difference with ref_contrast
        2) Creates new columns averaging performance across contrast
            for each method.
    Args:
        dataframe (pandas.Dataframe): Original dataframe 
        methods (list): list of method names
        contrasts (list): list of contrast names
        ref_contrast (str): reference contrast
        perf_suffix (str): suffix to use for performance column creation
    Returns:
        df_copy (pandas.Dataframe): Modifies dataframe contrast columns  
            with the PWD against the ref_contrast.
            Also, method performance columns are added following this 
            format [method_name]_perf_[measure_type]
        macro_perf_pwd_names (list): list of method performance 
            column names
    """
    df_copy = dataframe.copy()
    macro_perf_pwd_names = [method+perf_suffix for method in methods]

    for model_prefix in methods:
        agg_cols = [model_prefix+"_"+contrast for contrast in contrasts]
        model_ref_col = model_prefix+"_"+ref_contrast
        for model_contrast in agg_cols:
            #c_col = model_prefix+"_"+contrast
            df_copy[model_contrast] = 100 * (df_copy[model_ref_col] - 
                                             df_copy[model_contrast])/df_copy[model_ref_col]
        df_copy[model_prefix+perf_suffix] = np.mean([df_copy[col] for col in agg_cols], axis=0)
    
    return df_copy, macro_perf_pwd_names

def create_perf_df_sd(dataframe, methods, contrasts, perf_suffix="_perf_sd"):
    """Creates a copy of the original dataframe with added performance columns.
    Performance columns are patient-wise standard deviation of each method 
    across contrasts.
    Args:
        dataframe (pandas.Dataframe): Original dataframe containing
        methods (list): list of method names
        contrasts (list): list of all contrasts
        perf_suffix (str): suffix to use for performance column creation
    Returns:
        df_copy (pandas.Dataframe): Adds new performance columns to 
            dataframe following this format [method_name]_perf_[measure_type]
        macro_perf_sd_names (list): list of method performance column names
    """
    df_copy = dataframe.copy()
    macro_perf_sd_names = [method+perf_suffix for method in methods]
    
    for model_prefix in methods:
        agg_cols = [model_prefix+"_"+contrast for contrast in contrasts]
        df_copy[model_prefix+perf_suffix] = np.std([df_copy[col] for col in agg_cols], axis=0)
    
    return df_copy, macro_perf_sd_names

def contrast_specific_pwd_violin(dataframe, methods, contrasts, ref_contrast="t2w", outfile=None):
    """Plots several violin plots of each method representing
    the Pair-Wise Difference (PWD) between a contrast's CSA 
    and the reference contrast's CSA. The number of plots is 
    equal to the length of contrasts.
    Args:
        df (pandas.Dataframe): contains PWD values for each contrast
            across patients.
        methods (list): list of method names
        contrasts (list): contrast names excluding reference contrast
        ref_contrast (str): Reference contrast
    """
    quotient, remainder = divmod(len(contrasts), 2)
    nrows = quotient + 1 if remainder==1 else quotient
    cols_cat = ["#ff6767", "#8edba3"]
    
    fig, axs = plt.subplots(ncols=2, nrows=nrows, figsize=(12,nrows*8))
    filtered_axs = axs.reshape(-1)
    if remainder == 1:
        filtered_axs[-1].set_axis_off()        

    for contrast, ax in zip(contrasts, filtered_axs):
        models_contrast = [method+"_"+contrast for method in methods]
        cols_dic = {name: "#989e9a" if (k == 0 or k == 1) 
                else cols_cat[k%2] for k, name in enumerate(models_contrast)}
        sns.violinplot(data=dataframe[models_contrast], ax=ax, inner="box", palette=cols_dic)
        labels = models_contrast
        ax.set_title(f"{contrast} CSA % difference across methods w.r.t to {ref_contrast}")
        ax.xaxis.set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(0, len(labels)), labels, rotation=45, ha='right')
        ax.set_xlabel('Methods')
        ax.set_ylabel(f'% Difference in CSA values w.r.t {ref_contrast}')
        yabs_max = abs(max(ax.get_ylim(), key=abs))
        ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        
        bench_patch = mpatches.Patch(color="#989e9a", label='Benchmark')
        singleGT_patch = mpatches.Patch(color="#ff6767", label='Single GT')
        meanGT_patch = mpatches.Patch(color="#8edba3", label='Mean GT')
        ax.legend(title= "Method type", handles=[bench_patch, singleGT_patch, meanGT_patch])
        
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    
def macro_pwd_violin(df, methods, ref_contrast="t2w", outfile=None):
    """Plots a violin plot of each method representing the overall
    performance of a method across contrasts. This performance is
    measured via CSA pair-wise difference (PWD) w.r.t. a reference
    contrast.
    Args:
        df (pandas.Dataframe): contains PWD values for each contrast
            across patients.
        methods (list): method's performance column names following 
            this format [method_name]_perf_[measure_type] 
        ref_contrast (str): Reference contrast
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9,9))
    cols_cat = ["#ff6767", "#8edba3"]
    cols_dic = {name: "#989e9a" if (k == 0 or k == 1) else cols_cat[k%2] for k, name in enumerate(methods)}
    sns.violinplot(data=df[methods], ax=ax, inner="box", palette=cols_dic)

    labels = methods
    ax.set_title(" % Difference in CSA across all contrasts")
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, len(labels)), labels, rotation=45, ha='right')
    ax.set_xlabel('Methods')
    ax.set_ylabel(f'CSA % difference w.r.t {ref_contrast}')
    yabs_max = abs(max(ax.get_ylim(), key=abs))
    ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    bench_patch = mpatches.Patch(color="#989e9a", label='Benchmark')
    singleGT_patch = mpatches.Patch(color="#ff6767", label='Single GT')
    meanGT_patch = mpatches.Patch(color="#8edba3", label='Mean GT')
    ax.legend(title= "Method type", handles=[bench_patch, singleGT_patch, meanGT_patch])

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)


def macro_sd_violin_preli(df, methods, outfile=None):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
    cols_cat = ["#ff6767", "#8edba3"]
    cols_dic = {name: "#989e9a" if (k == 0 or k == 1) else cols_cat[k%2] for k, name in enumerate(methods)}

    sns.violinplot(data=df[methods], ax=ax, inner="box", palette=cols_dic)
    #labels = sd_macro_perf_names
    labels = ["hard_manual", "meanGT_manual", "meanGT_soft_singlecontrast", "meanGT_soft_allcontrast"]
    #labels = methods
    #labels = ["hard_manual", "meanGT_manual"]
    ax.set_title("CSA's standard deviation of the Ground Truth segmentations across contrasts", pad=20, fontweight="bold", fontsize="x-large")
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, len(labels)), labels, rotation=45, ha='right', fontsize="large")
    ax.set_xlabel('Ground Truth type', fontsize="x-large", fontweight="bold")
    ax.set_ylabel(f'Standard deviation ($mm^2$)', fontsize="x-large", fontweight="bold")


def macro_sd_violin(df, methods, outfile=None):  # HERE
    """Plots a violin plot of each method representing the overall
    performance of a method across contrasts. This performance is
    measured via the CSA\'s standard deviation across contrasts.
    Args:
        df (pandas.Dataframe): contains SD values for each contrast
            across patients.
        methods (list): method's performance column names following 
            this format [method_name]_perf_[measure_type]
    """
    sns.set_style('whitegrid', rc={'xtick.bottom': True,
                                 'ytick.left': True,})
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 9)) 
    plt.rcParams['legend.title_fontsize'] = 'x-large'
    plt.yticks(fontsize="x-large")
    
    cols_dic = {'manual_hard_GT_perf_sd': '#989e9a', 'manual_soft_GT_perf_sd': '#989e9a', 'hard_hard_perf_sd': '#ff6767', 'hard_soft_perf_sd': '#ff6767', 'meanGT_soft_perf_sd': '#8edba3', 'meanGT_soft_all_perf_sd': '#8edba3'}
    print(cols_dic)
    sns.violinplot(data=df[methods], ax=ax, inner="box", palette=cols_dic, linewidth=2)
    labels = ["GT Binary", "GT Average Soft", "(1) Binary GT - percontrast" ,"(2) Average soft GT – percontrast", "(3) Average soft  GT – all contrast"]

    ax.set_title("Variability of CSA across MRI contrasts", pad=20, fontweight="bold", fontsize=20)
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.grid(True, which='minor')
    ax.set_xticks(np.arange(0, len(labels)), labels, rotation=25, ha='right', fontsize=17)
    plt.yticks(fontsize=17)
    ax.tick_params(direction='out', axis='both')
    #ax.set_xlabel('Segmentation type', fontsize=17, fontweight="bold")
    ax.set_ylabel(r'Standard deviation ($\bf{mm^2}$)', fontsize=17, fontweight="bold")
    yabs_max = abs(max(ax.get_ylim(), key=abs))
    ax.set_ylim(ymax=(yabs_max + 2))
    

    # Here is the label and arrow code of interest
    ax.annotate('per contrast', xy=(0.6, 0.905), xytext=(0.6, 0.93), xycoords='axes fraction', 
            fontsize=15.0, ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=9.0, lengthB=1.0', lw=2.0, color='k'))
    ax.annotate('all contrasts', xy=(0.90, 0.905), xytext=(0.90, 0.93), xycoords='axes fraction', 
            fontsize=15.0, ha='center', va='bottom', color='black',
            arrowprops=dict(arrowstyle='-[, widthB=2.5, lengthB=1.0', lw=2.0, color='k'))


    bench_patch = mpatches.Patch(color="#989e9a", label='Manual GT')
    singleGT_patch = mpatches.Patch(color="#ff6767", label='Binary GT')
    meanGT_patch = mpatches.Patch(color="#8edba3", label='Average soft GT')
    ax.legend(#title= r"$\bf{Segmentation\  type}$", 
              handles=[bench_patch, singleGT_patch, meanGT_patch], 
              fontsize=14,
              loc="upper left", #, ncol=3, bbox_to_anchor=(0.5, 1.15), 
              frameon=True, 
              fancybox=True, 
              framealpha=0.7, 
              borderpad=1)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=400, bbox_inches="tight")

def create_experiment_folder():
    folder = "charts_" + str(datetime.now())
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder