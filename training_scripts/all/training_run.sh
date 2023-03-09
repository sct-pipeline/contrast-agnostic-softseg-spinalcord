####### Training Run #######                           
seeds=(15)
config_types=("hard_hard" "hard_soft" "meanGT_soft" "meanGT_hard")
contrasts=("T1w" "T2w" "T2star" "rec-average_dwi" "all")

for seed in ${seeds[@]}
do
  for config in ${config_types[@]}
  do
    for contrast in ${contrasts[@]}
    do
      sleep 5
      #echo ./batch_configs/"$config"_"$contrast"_seed="$seed".json
      output_dir=./results/miccai2023/"$config"_"$contrast"_seed="$seed"
      echo $output_dir
      mkdir $output_dir
      #echo ./batch_configs/"$config"_"$contrast"_seed="$seed".json
      CUDA_VISIBLE=2 ivadomed --train -c ./config/miccai2023/"$config"_"$contrast"_seed="$seed".json --path-output $output_dir
    done
  done
done