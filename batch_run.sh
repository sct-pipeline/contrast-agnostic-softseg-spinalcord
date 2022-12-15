####### Config creation #######

#python config_generator.py --config config_templates/hard_hard.json \
#                           --datasets /home/GRAMES.POLYMTL.CA/adelba/duke/projects/ivadomed/contrast-agnostic-seg/contrast-agnostic-centerofmass-preprocess-clean-all-2022-10-22/data_processed_clean \
#                           --ofolder joblibs/group-8 \
#                           --contrasts T1w T2w T2star rec-average_dwi \
#                           --seeds 15
#
#
#python config_generator.py --config config_templates/hard_soft.json \
#                           --datasets /home/GRAMES.POLYMTL.CA/adelba/duke/projects/ivadomed/contrast-agnostic-seg/contrast-agnostic-centerofmass-preprocess-clean-all-2022-10-22/data_processed_clean \
#                           --ofolder joblibs/group-8 \
#                           --contrasts T1w T2w T2star rec-average_dwi \
#                           --seeds 15                
       
python config_generator.py --config config_templates/meanGT_soft.json \
                           --datasets /home/GRAMES.POLYMTL.CA/adelba/duke/projects/ivadomed/contrast-agnostic-seg/contrast-agnostic-centerofmass-preprocess-clean-all-2022-10-22/data_processed_clean \
                           --ofolder joblibs/group-8 \
                           --contrasts T1w T2w T2star rec-average_dwi \
                           --seeds 15        

#python config_generator.py --config config_templates/meanGT_hard.json \
#                           --datasets /home/GRAMES.POLYMTL.CA/adelba/duke/projects/ivadomed/contrast-agnostic-seg/contrast-agnostic-centerofmass-preprocess-clean-all-2022-10-22/data_processed_clean \
#                           --ofolder joblibs/group-8 \
#                           --contrasts T1w T2w T2star rec-average_dwi \
#                           --seeds 15     

####### Training Run #######                           
seeds=(15) # seeds=(42 15 34 98 62)
config_types=("meanGT_soft") # ("hard_hard" "hard_soft" "meanGT_soft" "meanGT_hard")
contrasts=("T1w" "T2w" "T2star" "rec-average_dwi" "all")

for seed in ${seeds[@]}
do
  for config in ${config_types[@]}
  do
    for contrast in ${contrasts[@]}
    do
      sleep 5
      #echo ./batch_configs/"$config"_"$contrast"_seed="$seed".json
      output_dir=./batch_results/group-9/"$config"_"$contrast"_seed="$seed"
      echo $output_dir
      mkdir $output_dir
      #echo ./batch_configs/"$config"_"$contrast"_seed="$seed".json
      CUDA_VISIBLE=0 ivadomed --train -c ./batch_configs/group-9/"$config"_"$contrast"_seed="$seed".json --path-output $output_dir
    done
  done
done

          