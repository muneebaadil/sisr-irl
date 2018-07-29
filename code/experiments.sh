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

#2. ALMOST FINAL MODELS
#2.1. scale 4 
#VDSR 
#branch1
#python main.py --model VDSR --n_layers 20 --interpolate --patch_size 48 --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --enable_branches --half_resblocks --n_branches 1 --master_branch_pretrain ../experiment/model/VDSR.pt --loss 1*MSE --epochs 30 --save almost_final/VDSR_x4_b1 --reset
#branch2
#python main.py --model VDSR --n_layers 20 --interpolate --patch_size 48 --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --enable_branches --half_resblocks --n_branches 1 --pre_train ../experiment/almost_final/VDSR_x4_b1/model/model_best.pt --loss 1*MSE --epochs 30 --save almost_final/VDSR_x4_b2 --reset

#DRRN 
#branch1
#branch2

#LapSRN 
#branch1
#branch2

#MemNet 
#branch1
#branch2

#EDSRb 
#branch1 
#branch2

#2. Loss Tests
#2.1. EDSRb, scale 4
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*MSE --epochs 30 --save loss_tests/EDSRb_x4_MSE --reset --save_results --save_branches --save_residuals
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*L1 --epochs 30 --save loss_tests/EDSRb_x4_L1 --reset --save_results --save_branches --save_residuals
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 0.75*MSE+0.25*L1 --normalized_loss --epochs 30 --save loss_tests/EDSRb_x4_75MSE_25L1 --reset --save_results --save_branches --save_residuals
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*GradL2 --epochs 30 --normalized_loss --save loss_tests/EDSRb_x4_GradL2 --reset --save_results --save_branches --save_residuals
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 0.05*GradL2+0.95*MSE --normalized_loss --epochs 30 --save loss_tests/EDSRb_x4_05GradMSE_95MSE --reset --save_results --save_branches --save_residuals
#python main.py --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 0.1*GradL2+0.9*MSE --normalized_loss --epochs 30 --save loss_tests/EDSRb_x4_10GradMSE_90MSE --reset --save_results --save_branches --save_residuals
