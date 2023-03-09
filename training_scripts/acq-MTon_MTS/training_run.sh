####### Training Run #######                           
seeds=(15)
config_types=("hard_hard" "hard_soft" "meanGT_soft" "meanGT_hard")
contrasts=("flip-1_mt-on_MTS") # We can't differentiate between variants of MTS

for seed in ${seeds[@]}
do
  for config in ${config_types[@]}
  do
    for contrast in ${contrasts[@]}
    do
      sleep 5
      output_dir=./results/miccai2023/"$config"_"$contrast"_seed="$seed"
      echo $output_dir
      mkdir $output_dir
      CUDA_VISIBLE=0 ivadomed --train -c ./config/miccai2023/"$config"_"$contrast"_seed="$seed".json --path-output $output_dir
    done
  done
done