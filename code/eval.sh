#Set5

#LapSRN
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --pre_train ../experiment/model/LapSRN_x4.pt --save test/Set5/LapSRN_x4 --save_results --save_residuals --reset --test_only --data_test Set5
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/LapSRN_x4_b1/model/model_best.pt --save test/Set5/LapSRN_x4_b1 --save_results --save_residuals --reset --test_only --data_test Set5
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/LapSRN_x4_b2/model/model_best.pt --save test/Set5/LapSRN_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Set5

#SRResNet
#python main.py --model SRResNet --pre_train ../experiment/model/SRResNet_x4.pt --save test/Set5/SRResNet_x4 --save_results --save_residuals --reset --test_only --data_test Set5
#python main.py --model SRResNet --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/SRResNet_x4_b1/model/model_best.pt --save test/Set5/SRResNet_x4_b1 --save_results --save_residuals --reset --test_only --data_test Set5
#python main.py --model SRResNet --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/SRResNet_x4_b2/model/model_best.pt --save test/Set5/SRResNet_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Set5

#EDSRb 
#python main.py --model EDSR --pre_train ../experiment/model/EDSR_baseline_x4.pt --save test/Set5/EDSRb_x4 --save_results --save_residuals --reset --test_only --data_test Set5
#python main.py --model EDSR --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/EDSRb_x4_b1/model/model_best.pt --save test/Set5/EDSRb_x4_b1 --save_results --save_residuals --reset --test_only --data_test Set5
#python main.py --model EDSR --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/EDSRb_x4_b2/model/model_best.pt --save test/Set5/EDSRb_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Set5

#EDSR
#python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale .1 --pre_train ../experiment/model/EDSR_x4.pt --save test/Set5/EDSR_x4 --save_results --save_residuals --reset --test_only --data_test Set5
#python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale .1 --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/EDSR_x4_b1/model/model_best.pt --save test/Set5/EDSR_x4_b1 --save_results --save_residuals --reset --test_only --data_test Set5
#python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale .1 --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/EDSR_x4_b2/model/model_best.pt --save test/Set5/EDSR_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Set5

#RDN
#python main.py --model RDN --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --pre_train ../experiment/model/RDN_x4.pt --save test/Set5/RDN_x4 --save_results --save_residuals --reset --test_only --data_test Set5
#python main.py --model RDN --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --enable_branches --n_branches 1 --half_resblocks --pre_train /datadrive/sr-experiments/final_models/RDN_x4_b1/model/model_best.pt --save test/Set5/RDN_x4_b1 --save_results --save_residuals --reset --test_only --data_test Set5
#python main.py --model RDN --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --enable_branches --n_branches 2 --half_resblocks --pre_train /datadrive/sr-experiments/final_models/RDN_x4_b2/model/model_best.pt --save test/Set5/RDN_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Set5

#==============================================================
#==============================================================
#Set14

#LapSRN
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --pre_train ../experiment/model/LapSRN_x4.pt --save test/Set14/LapSRN_x4 --save_results --save_residuals --reset --test_only --data_test Set14
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/LapSRN_x4_b1/model/model_best.pt --save test/Set14/LapSRN_x4_b1 --save_results --save_residuals --reset --test_only --data_test Set14
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/LapSRN_x4_b2/model/model_best.pt --save test/Set14/LapSRN_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Set14

#SRResNet
#python main.py --model SRResNet --pre_train ../experiment/model/SRResNet_x4.pt --save test/Set14/SRResNet_x4 --save_results --save_residuals --reset --test_only --data_test Set14
#python main.py --model SRResNet --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/SRResNet_x4_b1/model/model_best.pt --save test/Set14/SRResNet_x4_b1 --save_results --save_residuals --reset --test_only --data_test Set14
#python main.py --model SRResNet --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/SRResNet_x4_b2/model/model_best.pt --save test/Set14/SRResNet_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Set14

#EDSRb 
#python main.py --model EDSR --pre_train ../experiment/model/EDSR_baseline_x4.pt --save test/Set14/EDSRb_x4 --save_results --save_residuals --reset --test_only --data_test Set14
#python main.py --model EDSR --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/EDSRb_x4_b1/model/model_best.pt --save test/Set14/EDSRb_x4_b1 --save_results --save_residuals --reset --test_only --data_test Set14
#python main.py --model EDSR --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/EDSRb_x4_b2/model/model_best.pt --save test/Set14/EDSRb_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Set14

#EDSR
#python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale .1 --pre_train ../experiment/model/EDSR_x4.pt --save test/Set14/EDSR_x4 --save_results --save_residuals --reset --test_only --data_test Set14
#python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale .1 --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/EDSR_x4_b1/model/model_best.pt --save test/Set14/EDSR_x4_b1 --save_results --save_residuals --reset --test_only --data_test Set14
#python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale .1 --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/EDSR_x4_b2/model/model_best.pt --save test/Set14/EDSR_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Set14

#RDN
#python main.py --model RDN --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --pre_train ../experiment/model/RDN_x4.pt --save test/Set14/RDN_x4 --save_results --save_residuals --reset --test_only --data_test Set14
#python main.py --model RDN --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --enable_branches --n_branches 1 --half_resblocks --pre_train /datadrive/sr-experiments/final_models/RDN_x4_b1/model/model_best.pt --save test/Set14/RDN_x4_b1 --save_results --save_residuals --reset --test_only --data_test Set14
#python main.py --model RDN --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --enable_branches --n_branches 2 --half_resblocks --pre_train /datadrive/sr-experiments/final_models/RDN_x4_b2/model/model_best.pt --save test/Set14/RDN_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Set14

#==============================================================
#==============================================================
#Urban100

#LapSRN
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --pre_train ../experiment/model/LapSRN_x4.pt --save test/Urban100/LapSRN_x4 --save_results --save_residuals --reset --test_only --data_test Urban100
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/LapSRN_x4_b1/model/model_best.pt --save test/Urban100/LapSRN_x4_b1 --save_results --save_residuals --reset --test_only --data_test Urban100
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/LapSRN_x4_b2/model/model_best.pt --save test/Urban100/LapSRN_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Urban100

#SRResNet
#python main.py --model SRResNet --pre_train ../experiment/model/SRResNet_x4.pt --save test/Urban100/SRResNet_x4 --save_results --save_residuals --reset --test_only --data_test Urban100
#python main.py --model SRResNet --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/SRResNet_x4_b1/model/model_best.pt --save test/Urban100/SRResNet_x4_b1 --save_results --save_residuals --reset --test_only --data_test Urban100
#python main.py --model SRResNet --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/SRResNet_x4_b2/model/model_best.pt --save test/Urban100/SRResNet_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Urban100

#EDSRb 
#python main.py --model EDSR --pre_train ../experiment/model/EDSR_baseline_x4.pt --save test/Urban100/EDSRb_x4 --save_results --save_residuals --reset --test_only --data_test Urban100
#python main.py --model EDSR --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/EDSRb_x4_b1/model/model_best.pt --save test/Urban100/EDSRb_x4_b1 --save_results --save_residuals --reset --test_only --data_test Urban100
#python main.py --model EDSR --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/EDSRb_x4_b2/model/model_best.pt --save test/Urban100/EDSRb_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Urban100

#EDSR
#python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale .1 --pre_train ../experiment/model/EDSR_x4.pt --save test/Urban100/EDSR_x4 --save_results --save_residuals --reset --test_only --data_test Urban100
#python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale .1 --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/EDSR_x4_b1/model/model_best.pt --save test/Urban100/EDSR_x4_b1 --save_results --save_residuals --reset --test_only --data_test Urban100
#python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale .1 --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/EDSR_x4_b2/model/model_best.pt --save test/Urban100/EDSR_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Urban100

#RDN
#python main.py --model RDN --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --pre_train ../experiment/model/RDN_x4.pt --save test/Urban100/RDN_x4 --save_results --save_residuals --reset --test_only --data_test Urban100
#python main.py --model RDN --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --enable_branches --n_branches 1 --half_resblocks --pre_train /datadrive/sr-experiments/final_models/RDN_x4_b1/model/model_best.pt --save test/Urban100/RDN_x4_b1 --save_results --save_residuals --reset --test_only --data_test Urban100
#python main.py --model RDN --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --enable_branches --n_branches 2 --half_resblocks --pre_train /datadrive/sr-experiments/final_models/RDN_x4_b2/model/model_best.pt --save test/Urban100/RDN_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test Urban100

#==============================================================
#==============================================================
#B100

#LapSRN
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --pre_train ../experiment/model/LapSRN_x4.pt --save test/B100/LapSRN_x4 --save_results --save_residuals --reset --test_only --data_test B100
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/LapSRN_x4_b1/model/model_best.pt --save test/B100/LapSRN_x4_b1 --save_results --save_residuals --reset --test_only --data_test B100
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/LapSRN_x4_b2/model/model_best.pt --save test/B100/LapSRN_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test B100

#SRResNet
#python main.py --model SRResNet --pre_train ../experiment/model/SRResNet_x4.pt --save test/B100/SRResNet_x4 --save_results --save_residuals --reset --test_only --data_test B100
#python main.py --model SRResNet --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/SRResNet_x4_b1/model/model_best.pt --save test/B100/SRResNet_x4_b1 --save_results --save_residuals --reset --test_only --data_test B100
#python main.py --model SRResNet --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/SRResNet_x4_b2/model/model_best.pt --save test/B100/SRResNet_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test B100

#EDSRb 
#python main.py --model EDSR --pre_train ../experiment/model/EDSR_baseline_x4.pt --save test/B100/EDSRb_x4 --save_results --save_residuals --reset --test_only --data_test B100
#python main.py --model EDSR --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/EDSRb_x4_b1/model/model_best.pt --save test/B100/EDSRb_x4_b1 --save_results --save_residuals --reset --test_only --data_test B100
#python main.py --model EDSR --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/EDSRb_x4_b2/model/model_best.pt --save test/B100/EDSRb_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test B100

#EDSR
#python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale .1 --pre_train ../experiment/model/EDSR_x4.pt --save test/B100/EDSR_x4 --save_results --save_residuals --reset --test_only --data_test B100
#python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale .1 --half_resblocks --enable_branches --n_branches 1 --pre_train /datadrive/sr-experiments/final_models/EDSR_x4_b1/model/model_best.pt --save test/B100/EDSR_x4_b1 --save_results --save_residuals --reset --test_only --data_test B100
#python main.py --model EDSR --n_resblocks 32 --n_feats 256 --res_scale .1 --half_resblocks --enable_branches --n_branches 2 --pre_train /datadrive/sr-experiments/final_models/EDSR_x4_b2/model/model_best.pt --save test/B100/EDSR_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test B100

#RDN
#python main.py --model RDN --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --pre_train ../experiment/model/RDN_x4.pt --save test/B100/RDN_x4 --save_results --save_residuals --reset --test_only --data_test B100
#python main.py --model RDN --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --enable_branches --n_branches 1 --half_resblocks --pre_train /datadrive/sr-experiments/final_models/RDN_x4_b1/model/model_best.pt --save test/B100/RDN_x4_b1 --save_results --save_residuals --reset --test_only --data_test B100
#python main.py --model RDN --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --enable_branches --n_branches 2 --half_resblocks --pre_train /datadrive/sr-experiments/final_models/RDN_x4_b2/model/model_best.pt --save test/B100/RDN_x4_b2 --save_results --save_residuals --save_branches --reset --test_only --data_test B100