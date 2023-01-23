####### Training Run #######                           
seeds=(15) # seeds=(42 15 34 98 62)
config_types=("hard_hard" "hard_soft" "meanGT_soft" "meanGT_hard")
#contrasts=("T1w" "T2w" "T2star" "rec-average_dwi" "all")
contrasts=("flip-2_mt-off_MTS") # We can't differentiate between variants of MTS

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
      CUDA_VISIBLE=1 ivadomed --train -c ./batch_configs/group-9/"$config"_"$contrast"_seed="$seed".json --path-output $output_dir
    done
  done
done