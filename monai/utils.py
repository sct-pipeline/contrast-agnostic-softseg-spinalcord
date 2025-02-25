import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn.functional as F
import subprocess
import os
import pandas as pd
import re
import importlib
import pkgutil
from image import Image


CONTRASTS = {
    "t1w": ["T1w", "space-other_T1w"],
    "t2w": ["T2w", "space-other_T2w"],
    "t2star": ["T2star", "space-other_T2star"],
    "dwi": ["rec-average_dwi", "acq-dwiMean_dwi"],
    "mt-on": ["flip-1_mt-on_space-other_MTS", "acq-MTon_MTR"],
    "mt-off": ["flip-2_mt-off_space-other_MTS"],
    "unit1": ["UNIT1"],
    "psir": ["PSIR"],
    "stir": ["STIR"]
}


def get_image_stats(image_path):
    """
    This function takes an image file as input and returns its orientation.

    Input:
        image_path : str : Path to the image file

    Returns:
        orientation : str : Orientation of the image
    """
    img = Image(str(image_path))
    img.change_orientation('RPI')
    shape = img.dim[:3]
    shape = [int(s) for s in shape]
    # Get pixdim
    pixdim = img.dim[4:7]
    # If all are the same, the image is isotropic
    if np.allclose(pixdim, pixdim[0], atol=1e-3):
        orientation = 'isotropic'
    # Elif, the lowest arg is 0 then the orientation is sagittal
    elif np.argmax(pixdim) == 0:
        orientation = 'sagittal'
    # Elif, the lowest arg is 1 then the orientation is coronal
    elif np.argmax(pixdim) == 1:
        orientation = 'coronal'
    # Else the orientation is axial
    else:
        orientation = 'axial'
    resolution = np.round(pixdim, 2)
    return shape, orientation, resolution


def get_pathology_wise_split(unified_df):
    
    # ===========================================================================
    #                Subject-wise Pathology split
    # ===========================================================================
    pathologies = unified_df['pathologyID'].unique()

    # count the number of subjects for each pathology
    pathology_subjects = {}
    for pathology in pathologies:
        pathology_subjects[pathology] = len(unified_df[unified_df['pathologyID'] == pathology]['subjectID'].unique())

    # merge MildCompression, DCM, MildCompression/DCM into DCM
    pathology_subjects['DCM'] = pathology_subjects['MildCompression'] + pathology_subjects['DCM'] + pathology_subjects['MildCompression/DCM']
    pathology_subjects.pop('MildCompression', None)
    pathology_subjects.pop('MildCompression/DCM', None)

    # ===========================================================================
    #                Contrast-wise Pathology split
    # ===========================================================================
    # for a given contrast, count the number of images for each pathology
    pathology_contrasts = {}
    for contrast in CONTRASTS.keys():
        pathology_contrasts[contrast] = {}
        # initialize the count for each pathology
        pathology_contrasts[contrast] = {pathology: 0 for pathology in pathologies}
        for pathology in pathologies:
                pathology_contrasts[contrast][pathology] += len(unified_df[(unified_df['pathologyID'] == pathology) & (unified_df['contrastID'] == contrast)]['filename'])

    # merge MildCompression, DCM, MildCompression/DCM into DCM
    for contrast in pathology_contrasts.keys():
        pathology_contrasts[contrast]['DCM'] = pathology_contrasts[contrast]['MildCompression'] + pathology_contrasts[contrast]['DCM'] + pathology_contrasts[contrast]['MildCompression/DCM']
        pathology_contrasts[contrast].pop('MildCompression', None)
        pathology_contrasts[contrast].pop('MildCompression/DCM', None)

    return pathology_subjects, pathology_contrasts
    

def plot_contrast_wise_pathology(df, path_save):
    # remove the TOTAL row
    df = df[:-1]
    # remove the #total_per_contrast column
    df = df.drop(columns=['#total_per_contrast'])

    color_palette = {
        'HC': '#55A868',
        'MS': '#2B4373',
        'RRMS': '#6A89C8',
        'PPMS': '#88A1E0',
        'RIS': '#4C72B0',
        'DCM': '#DD8452',
        'SCI': '#C44E52',
        'NMO': '#937860',
        'SYR': '#b3b3b3',
        'ALS': '#DA8BC3',
        'AcuteSCI': '#CCB974'
    }

    contrasts = df.index.tolist()

    # plot a pie chart for each contrast and save as different file
    # NOTE: some pathologies with less subjects were overlapping so this is a hacky (and bad) way to fix this 
    # issue temporarily by reordering the columns of the df
    for contrast in contrasts:
        df_contrast = df.loc[[contrast]].T
        # reorder the columsn to put 'ALS' between 'HC' and 'MS'
        if contrast in ['dwi']:
            df_contrast = df_contrast.reindex(['ALS', 'HC', 'MS', 'DCM', 'SCI', 'NMO', 'RRMS', 'PPMS', 'RIS', 'SYR'])
        elif contrast in ['t1w']:
            # reorder the columsn to put 'PPMS' between 'MS' and 'RRMS'
            df_contrast = df_contrast.reindex(['HC', 'MS', 'PPMS', 'RRMS', 'RIS', 'NMO', 'DCM', 'SCI', 'ALS', 'SYR'])
        elif contrast in ['t2star']:
            df_contrast = df_contrast.reindex(['HC', 'ALS', 'MS', 'DCM', 'SCI', 'NMO', 'RRMS', 'PPMS', 'RIS', 'SYR'])

        # for the given contrast, remove columns (pathologies) with 0 images
        df_contrast = df_contrast[df_contrast[contrast] != 0]
        
        # adapted from https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_and_donut_labels.html
        fig, ax = plt.subplots(figsize=(6.3, 3.5), subplot_kw=dict(aspect="equal"))  # Increased figure size
        wedges, texts = ax.pie(
            df_contrast[contrast], 
            wedgeprops=dict(width=0.5), 
            startangle=-40,
            colors=[color_palette[pathology] for pathology in df_contrast.index],
        )

        # Annotation customization
        bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
        texts_to_adjust = []  # collect all annotations for adjustment

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            # font size
            kw["fontsize"] = 14.5
            # bold font
            kw["fontweight"] = 'bold'

            # Push small labels further away from pie
            distance = 1.1
            # for dwi contrast and sci pathology, plot the annotation to the left
            if contrast == 'dwi' and df_contrast.index[i] == 'SCI':
                distance = 1.5
                horizontalalignment = 'right'
            if df_contrast.index[i] == 'ALS':
                distance = 1.2
                horizontalalignment = 'right'                
            if contrast == 't2w' and df_contrast.index[i] in ['RIS', 'ALS', 'PPMS']:
                if df_contrast.index[i] != 'PPMS':
                    distance = 1.4
                    horizontalalignment = 'left'
                else:
                    distance = 1
                    horizontalalignment = 'right'
            if df_contrast.index[i] == 'NMO':
                distance = 1.1
                horizontalalignment = 'right'
                
            # Annotate with number of images per pathology
            text = f"{df_contrast.index[i]} (n={df_contrast.iloc[i, 0]})"
            annotation = ax.annotate(text, xy=(x, y), xytext=(distance*np.sign(x)*1.05, distance*y),
                                     horizontalalignment=horizontalalignment, **kw)
            texts_to_adjust.append(annotation)

        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(path_save, f'{contrast}_pathology_split.png'), dpi=300)
        plt.close()


def parse_spacing(spacing_str):
    # Remove brackets and split by spaces
    spacing_values = re.findall(r"[\d.]+", spacing_str)
    # Convert to float
    return [float(val) for val in spacing_values]


def get_datasets_stats(datalists_root, contrasts_dict, path_save):

    # create a unified dataframe combining all datasets
    csvs = [os.path.join(datalists_root, file) for file in os.listdir(datalists_root) if file.endswith('_seed50.csv')]
    unified_df = pd.concat([pd.read_csv(csv) for csv in csvs], ignore_index=True)
    
    # sort the dataframe by the dataset column
    unified_df = unified_df.sort_values(by='datasetName', ascending=True)

    # save the originals as the csv
    unified_df.to_csv(os.path.join(path_save, 'dataset_contrast_agnostic.csv'), index=False)

    contrasts_final = list(contrasts_dict.keys())
    # rename the contrasts column as per contrasts_final
    for c in unified_df['contrastID'].unique():
        for cf in contrasts_final:
            if re.search(cf, c.lower()):
                unified_df.loc[unified_df['contrastID'] == c, 'contrastID'] = cf
                break

    # NOTE: MTon-MTR is same as flip-1_mt-on_space-other_MTS, but the naming is not mt-on
    # so doing the renaming manually
    unified_df.loc[unified_df['contrastID'] == 'acq-MTon_MTR', 'contrastID'] = 'mt-on'

    # convert 'spacing' column from a string like "[1. 1. 1.]" to a list of floats
    unified_df['spacing'] = unified_df['spacing'].apply(parse_spacing)
    # for contrast in contrasts_final:
    #     print(f"Max resolution for {contrast}:")
    #     print([unified_df[unified_df['contrastID'] == contrast]['spacing'].apply(lambda x: x[i]).max() for i in range(3)])

    splits = ['train', 'validation', 'test']
    # count the number of images per contrast
    df = pd.DataFrame(columns=['contrast', 'train', 'validation', 'test'])
    for contrast in contrasts_final:
        df.loc[len(df)] = [contrast, 0, 0, 0]
        # count the number of images per split
        for split in splits:
            df.loc[df['contrast'] == contrast, split] = len(unified_df[(unified_df['contrastID'] == contrast) & (unified_df['split'] == split)])

    # sort the dataframe by the contrast column
    df = df.sort_values(by='contrast', ascending=True)
    # add a row for the total number of images
    df.loc[len(df)] = ['TOTAL', df['train'].sum(), df['validation'].sum(), df['test'].sum()]
    # add a column for total number of images per contrast
    df['#images_per_contrast'] = df['train'] + df['validation'] + df['test']
    
    df_mega = pd.DataFrame()
    for orientation in ['sagittal', 'axial', 'isotropic']:
        
        # get the median resolutions per contrast
        df_res_median = pd.DataFrame(columns=[f'contrast', 'x', 'y', 'z'])
        df_res_min = pd.DataFrame(columns=[f'contrast', 'x', 'y', 'z'])
        df_res_max = pd.DataFrame(columns=[f'contrast', 'x', 'y', 'z'])
        df_res_mean = pd.DataFrame(columns=[f'contrast', 'x', 'y', 'z'])

        for contrast in contrasts_final:
            
            if len(unified_df[(unified_df['contrastID'] == contrast) & (unified_df['imgOrientation'] == orientation)]) == 0:
                # set NA values for the contrast
                df_res_median.loc[len(df_res_median)] = [f'{contrast}_{orientation}'] + [np.nan, np.nan, np.nan]
                df_res_min.loc[len(df_res_min)] = [f'{contrast}_{orientation}'] + [np.nan, np.nan, np.nan]
                df_res_max.loc[len(df_res_max)] = [f'{contrast}_{orientation}'] + [np.nan, np.nan, np.nan]
                df_res_mean.loc[len(df_res_mean)] = [f'{contrast}_{orientation}'] + [np.nan, np.nan, np.nan]
            
            else:
                # median
                df_res_median.loc[len(df_res_median)] = [f'{contrast}_{orientation}'] + [
                    unified_df[(unified_df['contrastID'] == contrast) & (unified_df['imgOrientation'] == orientation)
                               ]['spacing'].apply(lambda x: x[i]).median() for i in range(3)]
                # min
                df_res_min.loc[len(df_res_min)] = [f'{contrast}_{orientation}'] + [
                    unified_df[(unified_df['contrastID'] == contrast) & (unified_df['imgOrientation'] == orientation)
                               ]['spacing'].apply(lambda x: x[i]).min() for i in range(3)]
                # max
                df_res_max.loc[len(df_res_max)] = [f'{contrast}_{orientation}'] + [
                    unified_df[(unified_df['contrastID'] == contrast) & (unified_df['imgOrientation'] == orientation)
                               ]['spacing'].apply(lambda x: x[i]).max() for i in range(3)]
                # mean
                df_res_mean.loc[len(df_res_mean)] = [f'{contrast}_{orientation}'] + [
                    unified_df[(unified_df['contrastID'] == contrast) & (unified_df['imgOrientation'] == orientation)
                               ]['spacing'].apply(lambda x: x[i]).mean().round(2) for i in range(3)]

        # drop rows with NA values
        df_res_median = df_res_median.dropna()
        df_res_min = df_res_min.dropna()
        df_res_max = df_res_max.dropna()
        df_res_mean = df_res_mean.dropna()
        
        # combine the x,y,z columns into a single column and drop the x,y,z columns
        df_res_median['median_resolution_rpi'] = df_res_median.apply(lambda x: f"{x['x']} x {x['y']} x {x['z']}", axis=1)
        df_res_median = df_res_median.drop(columns=['x', 'y', 'z'])

        df_res_min['min_resolution_rpi'] = df_res_min.apply(lambda x: f"{x['x']} x {x['y']} x {x['z']}", axis=1)
        df_res_min = df_res_min.drop(columns=['x', 'y', 'z'])

        df_res_max['max_resolution_rpi'] = df_res_max.apply(lambda x: f"{x['x']} x {x['y']} x {x['z']}", axis=1)
        df_res_max = df_res_max.drop(columns=['x', 'y', 'z'])

        df_res_mean['mean_resolution_rpi'] = df_res_mean.apply(lambda x: f"{x['x']} x {x['y']} x {x['z']}", axis=1)
        df_res_mean = df_res_mean.drop(columns=['x', 'y', 'z'])

        # combine the dataframes based on the contrast column
        df_res = pd.merge(df_res_median, df_res_min, on='contrast')
        df_res = pd.merge(df_res, df_res_max, on='contrast')
        df_res = pd.merge(df_res, df_res_mean, on='contrast')

        # sort the dataframe by the contrast column
        df_res = df_res.sort_values(by='contrast', ascending=True)

        # concatenate the dataframes for different orientations on columns
        df_mega = pd.concat([df_mega, df_res], axis=0)
    
    # get the subject-wise pathology split
    pathology_subjects, pathology_contrasts = get_pathology_wise_split(unified_df)
    df_pathology = pd.DataFrame.from_dict(pathology_subjects, orient='index', columns=['Number of Subjects'])
    # rename index to Pathology
    df_pathology.index.name = 'Pathology'
    # sort the dataframe by the pathology column
    df_pathology = df_pathology.sort_index()
    # add a row for the total number of subjects
    df_pathology.loc['TOTAL'] = df_pathology['Number of Subjects'].sum()


    # get the contrast-wise pathology split
    df_contrast_pathology = pd.DataFrame.from_dict(pathology_contrasts, orient='index')
    # sort the dataframe by the contrast column
    df_contrast_pathology = df_contrast_pathology.sort_index()
    # add a row for the total number of images
    df_contrast_pathology.loc['TOTAL'] = df_contrast_pathology.sum()
    # add a column for the total number of images per contrast
    df_contrast_pathology['#total_per_contrast'] = df_contrast_pathology.sum(axis=1)
    # print(df_contrast_pathology)
    
    # plots
    save_path = os.path.join(path_save, 'plots')
    os.makedirs(save_path, exist_ok=True)
    plot_contrast_wise_pathology(df_contrast_pathology, save_path)
    # exit()

    # sort the csvs list
    csvs = sorted(csvs)

    # create a txt file
    with open(os.path.join(path_save, 'dataset_stats_overall.txt'), 'w') as f:
        # 1. write the datalists used in a bullet list
        f.write(f"DATASETS USED FOR MODEL TRAINING (n={len(csvs)}):\n")
        for csv in csvs:
            # only write the dataset name
            f.write(f"\t- {csv.split('_')[1]}\n")
        f.write("\n")

        # 2. write the subject-wise pathology split
        f.write(f"\nSUBJECT-WISE PATHOLOGY SPLIT:\n\n")
        f.write(df_pathology.to_markdown())
        f.write("\n\n\n")

        # 3. write the contrast-wise pathology split (a subject can have multiple contrasts)
        f.write(f"CONTRAST-WISE PATHOLOGY SPLIT (a subject can have multiple contrasts):\n\n")
        f.write(df_contrast_pathology.to_markdown())
        f.write("\n\n\n")

        # 4. write the train/validation/test split per contrast
        f.write(f"SPLITS ACROSS DIFFERENT CONTRASTS (n={len(contrasts_final)}):\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n\n")

        # 5. write the median, min, max and mean resolutions per contrast
        f.write(f"RESOLUTIONS PER CONTRAST PER ORIENTATION (in mm^3):\n\n")
        f.write(f"How to interpret the table: Each row corresponds to the contrast and its orientation with the median, min, max and mean resolutions in mm^3.\n")
        f.write(f"For simplification, if a contrast does not have any images in a particular orientation in the dataset, then the row is not present in the table.\n")
        f.write(f"For e.g. if you want to report the mean (min, max) resolution of a contrast, say, 'dwi_axial', \n")
        f.write(f"then you pick the respective element in each of the columns.\n")
        f.write(f"\t i.e. mean in-plane resolution: 0.89x0.89; range: (0.34x0.34, 1x1), and, likewise, Slice thickness: 5.1; range: (4, 17.5)\n\n")
        f.write(df_mega.to_markdown(index=False))
        f.write("\n\n")


def compute_gradcam(model, input_image, target_layer):
    """
    Computes Grad-CAM heatmap for 3D U-Net segmentation model.
    """
    # Ensure input is a tensor with gradients
    if hasattr(input_image, 'as_tensor'):
        input_image = input_image.as_tensor()
    
    # Create a new tensor that requires gradients
    input_tensor = input_image.clone().detach().requires_grad_(True)
    
    # Hooks for capturing activations and gradients
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    try:
        # Forward pass
        model.zero_grad()
        output = model(input_tensor)[0]
        
        # Ensure output is a standard tensor
        if hasattr(output, 'as_tensor'):
            output = output.as_tensor()
        
        # Create a tensor for gradient computation
        seg_output = output[0, 0]
        target_score = torch.sum(seg_output)  # Sum of all voxel activations
        
        # Backward pass with gradient computation
        model.zero_grad()
        target_score.backward()
        
        # Compute Grad-CAM weights
        grad_weights = torch.mean(gradients[0], dim=(2, 3, 4), keepdim=True)
        
        # Compute weighted activations
        gradcam_map = torch.sum(grad_weights * activations[0], dim=1).squeeze()
        
        # Normalize and apply ReLU
        gradcam_map = F.relu(gradcam_map)
        gradcam_map -= gradcam_map.min()
        gradcam_map /= (gradcam_map.max() + 1e-8)
        
        # Resize to input size
        gradcam_map = F.interpolate(
            gradcam_map.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode="trilinear",
            align_corners=False
        ).squeeze()
        
        return gradcam_map
    
    finally:
        forward_handle.remove()
        backward_handle.remove()

    return None


def visualize_feature_maps(input_img, model, layer_names, slice_axis=2, num_cols=4):
    """
    Visualize feature activation maps for specified layers of a 3D CNN.
    
    Args:
        model (torch.nn.Module): Trained CNN model
        nifti_path (str): Path to NIfTI image
        layer_names (list): Names of layers to visualize
        slice_axis (int): Axis to slice for 2D visualization (0, 1, or 2)
        num_cols (int): Number of columns in output visualization grid
    
    Returns:
        matplotlib figure with feature map visualizations
    """
    
    # convert input_img from a metaTensor to a tensor
    input_img = input_img.as_tensor()
    
    # Create feature extraction hook
    feature_maps = {}
    def hook_fn(module, input, output):
        feature_maps[module] = output.detach().cpu()
    
    # Register hooks for specified layers
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        model(input_img)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Create visualization
    fig, axes = plt.subplots(
        nrows=len(layer_names), 
        ncols=num_cols, 
        figsize=(4*num_cols, 4*len(layer_names))
    )
    
    for i, layer_name in enumerate(layer_names):
        # print(dict(model.named_modules()).keys())
        # Get feature maps for this layer
        layer_features = feature_maps.get(dict(model.named_modules())[layer_name], None)
        
        if layer_features is None:
            continue
        
        # Select subset of feature maps to visualize
        num_feature_maps = min(layer_features.shape[1], num_cols)
        
        for j in range(num_feature_maps):
            # Select middle slice along specified axis
            feature_slice = layer_features[0, j].numpy()
            mid_slice_idx = feature_slice.shape[slice_axis] // 2
            
            if slice_axis == 0:
                slice_2d = feature_slice[mid_slice_idx, :, :]
            elif slice_axis == 1:
                slice_2d = feature_slice[:, mid_slice_idx, :]
            else:  # slice_axis == 2
                slice_2d = feature_slice[:, :, mid_slice_idx]
            
            # Plot in appropriate subplot
            ax = axes[i, j] if len(layer_names) > 1 else axes[j]
            im = ax.imshow(slice_2d, cmap='viridis')
            ax.set_title(f'{layer_name} - Map {j}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    return fig


# Taken from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/find_class_by_name.py
def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(os.path.join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Check if any label image patch is empty in the batch
def check_empty_patch(labels):
    for i, label in enumerate(labels):
        if torch.sum(label) == 0.0:
            # print(f"Empty label patch found at index {i}. Skipping training step ...")
            return None
    return labels  # If no empty patch is found, return the labels


def get_git_branch_and_commit(dataset_path=None):
    """
    :return: git branch and commit ID, with trailing '*' if modified
    Taken from: https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/utils/sys.py#L476 
    and https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/utils/sys.py#L461
    """

    # branch info
    b = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=dataset_path)
    b_output, _ = b.communicate()
    b_status = b.returncode

    if b_status == 0:
        branch = b_output.decode().strip()
    else:
        branch = "!?!"

    # commit info
    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=dataset_path)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=dataset_path)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        unclean = True
        for line in output.decode().strip().splitlines():
            line = line.rstrip()
            if line.startswith("??"):  # ignore ignored files, they can't hurt
                continue
            break
        else:
            unclean = False
        if unclean:
            commit += "*"

    return branch, commit


def dice_score(prediction, groundtruth):
    smooth = 1.
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()
    # loss = (2 * numer + self.smooth) / (denor + self.smooth)
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice


def plot_slices(image, gt, pred, debug=False):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    # bring everything to numpy
    image = image.numpy()
    gt = gt.numpy()
    pred = pred.numpy()

    if not debug:
        mid_sagittal = image.shape[2]//2
        # plot X slices before and after the mid-sagittal slice in a grid
        fig, axs = plt.subplots(3, 6, figsize=(10, 6))
        fig.suptitle('Original Image --> Ground Truth --> Prediction')
        for i in range(6):
            axs[0, i].imshow(image[:, :, mid_sagittal-3+i].T, cmap='gray'); axs[0, i].axis('off') 
            axs[1, i].imshow(gt[:, :, mid_sagittal-3+i].T); axs[1, i].axis('off')
            axs[2, i].imshow(pred[:, :, mid_sagittal-3+i].T); axs[2, i].axis('off')

        # fig, axs = plt.subplots(1, 3, figsize=(10, 8))
        # fig.suptitle('Original Image --> Ground Truth --> Prediction')
        # slice = image.shape[2]//2

        # axs[0].imshow(image[:, :, slice].T, cmap='gray'); axs[0].axis('off') 
        # axs[1].imshow(gt[:, :, slice].T); axs[1].axis('off')
        # axs[2].imshow(pred[:, :, slice].T); axs[2].axis('off')
    
    else:   # plot multiple slices
        mid_sagittal = image.shape[2]//2
        # plot X slices before and after the mid-sagittal slice in a grid
        fig, axs = plt.subplots(3, 14, figsize=(20, 8))
        fig.suptitle('Original Image --> Ground Truth --> Prediction')
        for i in range(14):
            axs[0, i].imshow(image[:, :, mid_sagittal-7+i].T, cmap='gray'); axs[0, i].axis('off') 
            axs[1, i].imshow(gt[:, :, mid_sagittal-7+i].T); axs[1, i].axis('off')
            axs[2, i].imshow(pred[:, :, mid_sagittal-7+i].T); axs[2, i].axis('off')

    plt.tight_layout()
    fig.show()
    return fig


class PolyLRScheduler(_LRScheduler):
    """
    Polynomial learning rate scheduler. Taken from:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/lr_scheduler/polylr.py

    """

    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


if __name__ == "__main__":

    # seed = 54
    # num_cv_folds = 10
    # names_list = FoldGenerator(seed, num_cv_folds, 100).get_fold_names()
    # tr_ix, val_tx, te_ix, fold = names_list[0]
    # print(len(tr_ix), len(val_tx), len(te_ix))

    datalists_root = "/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/datalists/20250115-v21PtrAll"
    get_datasets_stats(datalists_root, contrasts_dict=CONTRASTS, path_save=datalists_root)
    # get_pathology_wise_split(datalists_root, path_save=datalists_root)

    # img_path = "/home/GRAMES.POLYMTL.CA/u114716/datasets/sci-colorado/sub-5694/anat/sub-5694_T2w.nii.gz"
    # get_image_stats(img_path)