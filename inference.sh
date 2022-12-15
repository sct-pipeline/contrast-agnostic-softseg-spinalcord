seeds=(15) # seeds=(42 15 34 98 62)
config_types=("hard_hard" "hard_soft" "meanGT_soft" "meanGT_hard")
contrasts=("T1w" "T2star" "T2w" "rec-average_dwi" "all")

for seed in ${seeds[@]}
do
  for config in ${config_types[@]}
  do
    for contrast in ${contrasts[@]}
    do
      sleep 5
      output_dir=../duke/temp/adrian/contrast-agnostic-seg-models/Group8_01-12-2022/"$config"_"$contrast"_seed="$seed"/
      echo $output_dir
      CUDA_VISIBLE=0 ivadomed --test -c ../duke/temp/adrian/contrast-agnostic-seg-models/Group8_01-12-2022/"$config"_"$contrast"_seed="$seed"/config_file.json --path-output $output_dir
    done
  done
done
