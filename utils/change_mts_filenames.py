"""
python config_generator.py --config config_templates/hard_hard.json \
                           --datasets /home/GRAMES.POLYMTL.CA/adelba/duke/projects/ivadomed/contrast-agnostic-seg/contrast-agnostic-centerofmass-preprocess-clean-all-2022-10-22/data_processed_clean \
                           --ofolder joblibs/MTS/T1w \
                           --contrasts T1w T2w T2star rec-average_dwi flip-2_mt-off_MTS flip-1_mt-on_MTS \
                           --seeds 15

python config_generator.py --config config_templates/hard_soft.json \
                           --datasets /home/GRAMES.POLYMTL.CA/adelba/duke/projects/ivadomed/contrast-agnostic-seg/contrast-agnostic-centerofmass-preprocess-clean-all-2022-10-22/data_processed_clean \
                           --ofolder joblibs/MTS/T1w \
                           --contrasts T1w T2w T2star rec-average_dwi flip-2_mt-off_MTS flip-1_mt-on_MTS \
                           --seeds 15

python config_generator.py --config config_templates/meanGT_soft.json \
                           --datasets /home/GRAMES.POLYMTL.CA/adelba/duke/temp/adrian/contrast-agnostic-seg-models/data_processed_clean \
                           --ofolder joblibs/MTS/all \
                           --contrasts T1w T2w T2star rec-average_dwi flip-1_mt-on_MTS flip-2_mt-off_MTS \
                           --seeds 15

python config_generator.py --config config_templates/meanGT_hard.json \
                           --datasets /home/GRAMES.POLYMTL.CA/adelba/duke/projects/ivadomed/contrast-agnostic-seg/contrast-agnostic-centerofmass-preprocess-clean-all-2022-10-22/data_processed_clean \
                           --ofolder joblibs/MTS/T1w \
                           --contrasts T1w T2w T2star rec-average_dwi flip-2_mt-off_MTS flip-1_mt-on_MTS \
                           --seeds 15     
"""
import os 
from glob import glob

if __name__ == "__main__":
    user = "adelba" # POLYGRAMS
    user_temp_folder = "adrian" # DUKE TEMP FOLDER NAME
    dataset_type = "data_processed_clean"

    dir_base = f"/home/GRAMES.POLYMTL.CA/{user}/duke/temp/{user_temp_folder}/contrast-agnostic-seg-models/{dataset_type}/**/**/*"
    dir_labels = f"/home/GRAMES.POLYMTL.CA/{user}/duke/temp/{user_temp_folder}/contrast-agnostic-seg-models/{dataset_type}/derivatives/labels/**/**/*"
    dir_soft_labels = f"/home/GRAMES.POLYMTL.CA/{user}/duke/temp/{user_temp_folder}/contrast-agnostic-seg-models/{dataset_type}/derivatives/labels_softseg/**/**/*"

    files_base = [fn for fn in glob(dir_base) if os.path.isfile(fn)]
    files_labels = [fn for fn in glob(dir_labels) if os.path.isfile(fn)]
    files_soft_labels = [fn for fn in glob(dir_soft_labels) if os.path.isfile(fn)]

    for dr in [files_base, files_labels, files_soft_labels]:
        for f in dr:
            base_path = os.path.basename(f)
            if "acq-MTon_MTS" in base_path:
                new_filename = f.replace("acq-MTon_MTS", "flip-1_mt-on_MTS")
                assert dataset_type != "data_processed_clean_T1w_MTS"
                os.rename(f, new_filename)
            elif "acq-T1w_MTS" in base_path:
                new_filename = f.replace("acq-T1w_MTS", "flip-2_mt-off_MTS")
                assert dataset_type != "data_processed_clean_MTon_MTS"
                os.rename(f, new_filename)
            else:
                continue
            print(f"Replaced : {base_path} --> {os.path.basename(new_filename)}")
