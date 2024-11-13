import os, re
import numpy as np
import pandas as pd


CONTRASTS = {
    "t1map": ["T1map"],
    "mp2rage": ["inv-1_part-mag_MP2RAGE", "inv-2_part-mag_MP2RAGE"],
    "t1w": ["T1w", "space-other_T1w", "acq-lowresSag_T1w"],
    "t2w": ["T2w", "space-other_T2w", "acq-lowresSag_T2w", "acq-highresSag_T2w"],
    "t2star": ["T2star", "space-other_T2star"],
    "dwi": ["rec-average_dwi", "acq-dwiMean_dwi"],
    "mt-on": ["flip-1_mt-on_space-other_MTS", "acq-MTon_MTR"],
    "mt-off": ["flip-2_mt-off_space-other_MTS"],
    "unit1": ["UNIT1"],
    "psir": ["PSIR"],
    "stir": ["STIR"]
}


def generate_table(df, path_save):
    contrast_stats = {}

    # replace 'coronal' with 'axial' in imgOrientation
    df.loc[df['imgOrientation'] == 'coronal', 'imgOrientation'] = 'axial'

    # get unique imgOrientations
    img_orientations = df['imgOrientation'].unique()

    for contrast in CONTRASTS.keys():
        contrast_stats[contrast] = {}

        for orientation in img_orientations:            
            contrast_stats[contrast][orientation] = {"n": 0, "spacing_min": [], "spacing_max": [], "size_min": [], "size_max": []}
            
            # get the number of images with the contrast and orientation
            contrast_stats[contrast][orientation]['n'] = len(df[(df['contrastID'] == contrast) & (df['imgOrientation'] == orientation)])

            if contrast_stats[contrast][orientation]['n'] == 0:
                # if there are no images with this contrast and orientation, remove the key
                del contrast_stats[contrast][orientation]
                continue

            # create a temp list to store the spacings and sizes of the images
            all_spacings, all_sizes = [], []

            for i in range(contrast_stats[contrast][orientation]['n']):
                # get the spacings and sizes of the images
                all_spacings.append(df[(df['contrastID'] == contrast) & (df['imgOrientation'] == orientation)]['spacing'].iloc[i])
                all_sizes.append(df[(df['contrastID'] == contrast) & (df['imgOrientation'] == orientation)]['shape'].iloc[i])

            # Convert the list of strings to a numpy array
            all_spacings = np.array([np.fromstring(s.strip('[]'), sep=' ') for s in all_spacings])
            all_sizes = np.array([np.fromstring(s.strip('[]'), sep=',') for s in all_sizes])

            # get the min and max spacings across the respective dimensions
            contrast_stats[contrast][orientation]['spacing_min'] = np.min(all_spacings, axis=0)
            contrast_stats[contrast][orientation]['spacing_max'] = np.max(all_spacings, axis=0)

            # get the min and max sizes across the respective dimensions
            contrast_stats[contrast][orientation]['size_min'] = np.min(all_sizes, axis=0)
            contrast_stats[contrast][orientation]['size_max'] = np.max(all_sizes, axis=0)

    # create a dataframe from contrast_stats
    df_img_stats = pd.DataFrame.from_dict({(i, j): contrast_stats[i][j]
                                             for i in contrast_stats.keys()
                                                for j in contrast_stats[i].keys()},)
    df_img_stats = df_img_stats.T
    print(df_img_stats)

            

def main(datalists_root, contrasts_dict, path_save):

    # create a unified dataframe combining all datasets
    csvs = [os.path.join(datalists_root, file) for file in os.listdir(datalists_root) if file.endswith('_seed50.csv')]
    unified_df = pd.concat([pd.read_csv(csv) for csv in csvs], ignore_index=True)
    
    # sort the dataframe by the dataset column
    unified_df = unified_df.sort_values(by='datasetName', ascending=True)

    # dropna
    unified_df = unified_df.dropna(subset=['pathologyID'])

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

    generate_table(unified_df, path_save)



if __name__ == "__main__":

    datalists_root = "/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/datalists/v2-final-aggregation-20241022"
    main(datalists_root, CONTRASTS, datalists_root)
