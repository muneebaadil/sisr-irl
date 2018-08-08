#DATASET SETUP 
#python main.py --model EDSR --scale 4 --save EDSR_baseline_x2 --reset --ext sep_reset --train_only --dir_data [paste-data-dir-here]

#0. DIRECT VS RESIDUAL LEARNING EXPERIMENTS 
#0.1. EDSRb
#0.1.1. scale 4 
#python main.py --model EDSR --enable_branches --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*MSE --epochs 30 --save learning_label/EDSRb_x4_residual_MSE --reset 
#python main.py --model EDSR --enable_branches --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*L1 --epochs 30 --save learning_label/EDSRb_x4_residual_L1 --reset 
#python main.py --model EDSR --enable_branches --n_branches 1 --branch_label hr --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*MSE --epochs 30 --save learning_label/EDSRb_x4_hr_MSE --reset 
#python main.py --model EDSR --enable_branches --n_branches 1 --branch_label hr --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*L1 --epochs 30 --save learning_label/EDSRb_x4_hr_L1 --reset 

#0.2. LapSRN
#0.2.1. Scale 4
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_feats 64 --n_layers 10 --enable_branches --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/LapSRN_x4.pt --loss 1*MSE --epochs 30 --save learning_label/LapSRN_x4_residual_MSE --reset 
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_feats 64 --n_layers 10 --enable_branches --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/LapSRN_x4.pt --loss 1*L1 --epochs 30 --save learning_label/LapSRN_x4_residual_L1 --reset 
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_feats 64 --n_layers 10 --enable_branches --n_branches 1 --branch_label hr --master_branch_pretrain ../experiment/model/LapSRN_x4.pt --loss 1*MSE --epochs 30 --save learning_label/LapSRN_x4_hr_MSE --reset 
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_feats 64 --n_layers 10 --enable_branches --n_branches 1 --branch_label hr --master_branch_pretrain ../experiment/model/LapSRN_x4.pt --loss 1*L1 --epochs 30 --save learning_label/LapSRN_x4_hr_L1 --reset 

#1. DOWNSAMPLED FEATURE MAPS EXPERIMENTS
#1.1. scale 4 
#up version already done at section 0.1.1.
#python main.py --model EDSR --enable_branches --n_branches 1 --branch_label residual --down_feats --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*MSE --epochs 30 --save down_feats/EDSRb_x4_down --reset 

#up version already done at section 0.2.1.
#python main.py --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_feats 64 --n_layers 10 --enable_branches --n_branches 1 --branch_label residual --down_feats --master_branch_pretrain ../experiment/model/LapSRN_x4.pt --loss 1*MSE --epochs 30 --save down_feats/LapSRN_x4_down --reset 

#python main.py --model SRResNet --enable_branches --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/SRResNet_x4.pt --loss 1*MSE --epochs 30 --save down_feats/SRResNet_x4_up --reset
#python main.py --model SRResNet --enable_branches --n_branches 1 --branch_label residual --down_feats --master_branch_pretrain ../experiment/model/SRResNet_x4.pt --loss 1*MSE --epochs 30 --save down_feats/SRResNet_x4_down --reset

#2. ALMOST FINAL MODELS (x4 scale)
#VDSR 
#branch1
#python main.py --model VDSR --interpolate --patch_size 42 --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 20 --enable_branches --n_branches 1 --master_branch_pretrain ../experiment/model/VDSR.pt --loss 1*MSE --epochs 30 --save almost_final/VDSR_x4_b1 --reset
#branch2
#python main.py --model VDSR --interpolate --patch_size 42 --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 20 --enable_branches --n_branches 2 --pre_train ../experiment/almost_final/VDSR_x4_b1/model/model_best.pt --loss 1*MSE --epochs 30 --save almost_final/VDSR_x4_b2 --reset

#DRRN 
#branch1
#python main.py --model DRRN --interpolate --patch_size 32 --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 25 --n_feats 128 --enable_branches --n_branches 1 --master_branch_pretrain ../experiment/model/DRRN.pt --loss 1*MSE --epochs 30 --save almost_final/DRRN_x4_b1 --reset
#branch2
#python main.py --model DRRN --interpolate --patch_size 32 --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 25 --n_feats 128 --enable_branches --n_branches 2 --pre_train ../experiment/almost_final/DRRN_x4_b1/model/model_best.pt --loss 1*MSE --epochs 30 --save almost_final/DRRN_x4_b2 --reset

#MemNet 
#branch1
#branch2

#LapSRN 
#branch1
#python main.py --model LapSRN --patch_size 128 --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --half_resblocks --enable_branches --n_branches 1 --master_branch_pretrain ../experiment/model/LapSRN_x4.pt --loss 1*L1 --epochs 30 --save almost_final/LapSRN_x4_b1 --reset
#branch2
#python main.py --model LapSRN --patch_size 128 --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --half_resblocks --enable_branches --n_branches 2 --pre_train ../experiment/almost_final/LapSRN_x4_b1/model/model_best.pt --loss 1*L1 --epochs 30 --save almost_final/LapSRN_x4_b2 --reset

#EDSRb 
#branch1
#python main.py --model EDSR --patch_size 96 --enable_branches --n_branches 1 --half_resblocks --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*L1 --epochs 30 --save almost_final/EDSRb_x4_b1 --reset  
#branch2
#python main.py --model EDSR --patch_size 96 --enable_branches --n_branches 2 --half_resblocks --pre_train ../experiment/almost_final/EDSRb_x4_b1/model/model_best.pt --loss 1*L1 --epochs 30 --save almost_final/EDSRb_x4_b2 --reset  

#EDSR
#branch1
#python main.py --model EDSR --patch_size 96 --n_resblocks 32 --n_feats 256 --res_scale .1 --enable_branches --n_branches 1 --half_resblocks --master_branch_pretrain ../experiment/model/EDSR_x4.pt --loss 1*L1 --epochs 30 --save almost_final/EDSR_x4_b1 --reset  
#branch2
#python main.py --model EDSR --patch_size 96 --n_resblocks 32 --n_feats 256 --res_scale .1 --enable_branches --n_branches 2 --half_resblocks --pre_train ../experiment/almost_final/EDSRb_x4_b1/model/model_best.pt --loss 1*L1 --epochs 30 --save almost_final/EDSR_x4_b2 --reset  

#RDN
#branch1
#python main.py --model RDN --patch_size 128 --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --enable_branches --n_branches 1 --half_resblocks --master_branch_pretrain ../experiment/model/RDN_x4.pt --loss 1*L1 --epochs 30 --save almost_final/RDN_x4_b1 --reset  
#branch2
#python main.py --model RDN --patch_size 128 --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --enable_branches --n_branches 2 --half_resblocks --pre_train ../experiment/almost_final/RDN_x4_b1/model/model_best.pt --loss 1*L1 --epochs 30 --save almost_final/RDN_x4_b2 --reset  

#2. Loss Tests
#2.1. EDSRb, scale 4
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*MSE --epochs 30 --save loss_tests/EDSRb_x4_MSE --reset --save_branches
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*L1 --epochs 30 --save loss_tests/EDSRb_x4_L1 --reset --save_branches
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 0.75*MSE+0.25*L1 --normalized_loss --epochs 30 --save loss_tests/EDSRb_x4_75MSE_25L1 --reset --save_branches
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*GradL2 --epochs 30 --normalized_loss --save loss_tests/EDSRb_x4_GradL2 --reset --save_branches
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 0.05*GradL2+0.95*MSE --normalized_loss --epochs 30 --save loss_tests/EDSRb_x4_05GradMSE_95MSE --reset --save_branches
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 0.1*GradL2+0.9*MSE --normalized_loss --epochs 30 --save loss_tests/EDSRb_x4_10GradMSE_90MSE --reset --save_branches
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*SSIM --normalized_loss --epochs 30 --save loss_tests/ssim --reset --save_branches
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*MSSSIM --normalized_loss --epochs 30 --save loss_tests/msssim --reset --save_branches

#3. FINAL MODELS
#3.1. Scale 2 

#3.2. Scale 3 

#3.4. Scale 4 
#RDN
#python main.py --model RDN --patch_size 128 --n_denseblocks 16 --n_layers 8 --n_feats 64 --growth_rate 64 --enable_branches --n_branches 1 --half_resblocks --master_branch_pretrain ../experiment/model/RDN_x4.pt --loss 1*MSE --epochs 30 --batch_size 8 --save final_models/RDN_x4_b1 --reset --chop
#EDSR
#python main.py --model EDSR --patch_size 96 --n_resblocks 32 --n_feats 256 --res_scale .1 --enable_branches --n_branches 1 --half_resblocks --master_branch_pretrain ../experiment/model/EDSR_x4.pt --loss 1*MSE --epochs 30 --save final_models/EDSR_x4_b1 --reset --chop
#EDSRb
#python main.py --model EDSR --patch_size 96 --enable_branches --n_branches 1 --half_resblocks --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*MSE --epochs 50 --save final_models/EDSRb_x4_b1 --reset
#LapSRN
#python main.py --model LapSRN --patch_size 128 --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_layers 10 --n_feats 64 --half_resblocks --enable_branches --n_branches 1 --master_branch_pretrain ../experiment/model/LapSRN_x4.pt --loss 1*MSE --epochs 30 --save final/LapSRN_x4_b1 --reset

#4. GAN extension
