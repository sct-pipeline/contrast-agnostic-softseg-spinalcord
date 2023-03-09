####### Config creation #######
## Specific the contrasts you would like to consider during your experiments. 
## You would need to create the config file for each one of your config 
## template the following way:


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
