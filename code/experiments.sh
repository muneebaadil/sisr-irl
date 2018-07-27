#0. DIRECT VS RESIDUAL LEARNING EXPERIMENTS 
#0.1. EDSRb
#0.1.1. scale 4 
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model EDSR --enable_branches --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*MSE --epochs 30 --save learning_label/EDSRb_x4_residual_MSE --reset 
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model EDSR --enable_branches --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*L1 --epochs 30 --save learning_label/EDSRb_x4_residual_L1 --reset 
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model EDSR --enable_branches --n_branches 1 --branch_label hr --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*MSE --epochs 30 --save learning_label/EDSRb_x4_hr_MSE --reset 
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model EDSR --enable_branches --n_branches 1 --branch_label hr --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*L1 --epochs 30 --save learning_label/EDSRb_x4_hr_L1 --reset 

#0.2. LapSRN
#0.2.1. Scale 4
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_feats 64 --n_layers 10 --enable_branches --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/LapSRN_x4.pt --loss 1*MSE --epochs 30 --save learning_label/LapSRN_x4_residual_MSE --reset 
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_feats 64 --n_layers 10 --enable_branches --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/LapSRN_x4.pt --loss 1*L1 --epochs 30 --save learning_label/LapSRN_x4_residual_L1 --reset 
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_feats 64 --n_layers 10 --enable_branches --n_branches 1 --branch_label hr --master_branch_pretrain ../experiment/model/LapSRN_x4.pt --loss 1*MSE --epochs 30 --save learning_label/LapSRN_x4_hr_MSE --reset 
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_feats 64 --n_layers 10 --enable_branches --n_branches 1 --branch_label hr --master_branch_pretrain ../experiment/model/LapSRN_x4.pt --loss 1*L1 --epochs 30 --save learning_label/LapSRN_x4_hr_L1 --reset 

#1. DOWNSAMPLED FEATURE MAPS EXPERIMENTS
#1.1. scale 4 
#up version already done at section 0.1.1.
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model EDSR --enable_branches --n_branches 1 --branch_label residual --down_feats --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*MSE --epochs 30 --save down_feats/EDSRb_x4_down --reset 

#up version already done at section 0.2.1.
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_feats 64 --n_layers 10 --enable_branches --n_branches 1 --branch_label residual --down_feats --master_branch_pretrain ../experiment/model/LapSRN_x4.pt --loss 1*MSE --epochs 30 --save down_feats/LapSRN_x4_down --reset 

#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model SRResNet --enable_branches --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/SRResNet_x4.pt --loss 1*MSE --epochs 30 --save down_feats/SRResNet_x4_down --reset
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model SRResNet --enable_branches --n_branches 1 --branch_label residual --down_feats --master_branch_pretrain ../experiment/model/SRResNet_x4.pt --loss 1*MSE --epochs 30 --save down_feats/SRResNet_x4_down --reset

#1.2. scale 3 
#python main.py --dir_data /datadrive/ --ext sep --scale 3 --model EDSR --enable_branches --half_resblocks --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/EDSR_baseline_x3.pt --loss 1*MSE --epochs 30 --save down_feats/EDSRb_x3_up --reset 
#python main.py --dir_data /datadrive/ --ext sep --scale 3 --model EDSR --enable_branches --half_resblocks --n_branches 1 --branch_label residual --down_feats --master_branch_pretrain ../experiment/model/EDSR_baseline_x3.pt --loss 1*MSE --epochs 30 --save down_feats/EDSRb_x3_down --reset 

#1.3. scale 2 
#python main.py --dir_data /datadrive/ --ext sep --scale 2 --model EDSR --enable_branches --half_resblocks --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/EDSR_baseline_x2.pt --loss 1*MSE --epochs 30 --save down_feats/EDSRb_x4_up --reset 
#python main.py --dir_data /datadrive/ --ext sep --scale 2 --model EDSR --enable_branches --half_resblocks --n_branches 1 --branch_label residual --down_feats --master_branch_pretrain ../experiment/model/EDSR_baseline_x2.pt --loss 1*MSE --epochs 30 --save down_feats/EDSRb_x4_down --reset 
#python main.py --dir_data /datadrive/ --ext sep --scale 2 --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_feats 64 --n_layers 10 --enable_branches --half_resblocks --n_branches 1 --branch_label residual --master_branch_pretrain ../experiment/model/LapSRN_x2.pt --loss 1*MSE --epochs 30 --save down_feats/LapSRN_x2_up --reset
#python main.py --dir_data /datadrive/ --ext sep --scale 2 --model LapSRN --rgb_range 1 --n_channel_in 1 --n_channel_out 1 --n_feats 64 --n_layers 10 --enable_branches --half_resblocks --n_branches 1 --branch_label residual --down_feats --master_branch_pretrain ../experiment/model/LapSRN_x2.pt --loss 1*MSE --epochs 30 --save down_feats/LapSRN_x2_down --reset

#2. Loss Tests
#2.1. EDSRb, scale 4
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*MSE --epochs 30 --save loss_tests/EDSRb_x4_MSE --reset --save_results --save_branches --save_residuals
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*L1 --epochs 30 --save loss_tests/EDSRb_x4_L1 --reset --save_results --save_branches --save_residuals
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 0.75*MSE+0.25*L1 --normalized_loss --epochs 30 --save loss_tests/EDSRb_x4_75MSE_25L1 --reset --save_results --save_branches --save_residuals
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 1*GradL2 --epochs 30 --normalized_loss --save loss_tests/EDSRb_x4_GradL2 --reset --save_results --save_branches --save_residuals
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 0.05*GradL2+0.95*MSE --normalized_loss --epochs 30 --save loss_tests/EDSRb_x4_05GradMSE_95MSE --reset --save_results --save_branches --save_residuals
#python main.py --dir_data /datadrive/ --ext sep --scale 4 --model EDSR --enable_branches --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --loss 0.1*GradL2+0.9*MSE --normalized_loss --epochs 30 --save loss_tests/EDSRb_x4_10GradMSE_90MSE --reset --save_results --save_branches --save_residuals
