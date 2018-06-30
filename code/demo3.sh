#half_both runs
#scale 4
#python main.py --model EDSR --scale 4 --enable_rrl --half_resblocks --half_feats --n_branches 1 --reset --master_branch_pretrain ../experiment/model/EDSR_baseline_x4.pt --dir_data /datadrive/ --ext bin --print_model --n_val 10 --epochs 30 --save half_tests_new/half_both/EDSR_baseline_x4_b1
#python main.py --model EDSR --scale 4 --enable_rrl --half_resblocks --half_feats --n_branches 2 --reset --pretrain ../experiment/half_tests_new/half_both/EDSR_baseline_x4_b1/ --dir_data /datadrive/ --ext bin --print_model --n_val 10 --epochs 30 --save half_tests_new/half_both/EDSR_baseline_x4_b2

#scale 3 
#python main.py --model EDSR --scale 3 --enable_rrl --half_resblocks --half_feats --reset --pre_train ../experiment/model/EDSR_baseline_x3.pt --dir_data /datadrive/ --ext bin --print_model --n_val 10 --epochs 30 --save test_x3

#scale 2
#python main.py --model EDSR --scale 2 --enable_rrl --half_resblocks --half_feats --reset --pre_train ../experiment/model/EDSR_baseline_x2.pt --dir_data /datadrive/ --ext bin --print_model --n_val 10 --epochs 30 --save test_x2

#loss_test runs 
#python main.py --model EDSR --scale 4 --enable_rrl --half_resblocks --half_feats --reset --pre_train ../experiment/model/EDSR_baseline_x4.pt --dir_data /datadrive/ --ext bin --print_model --n_val 10 --epochs 30 --save loss_tests/charbonnier --loss 1*Charbonnier
#python main.py --model EDSR --scale 4 --enable_rrl --half_resblocks --half_feats --reset --pre_train ../experiment/model/EDSR_baseline_x4.pt --dir_data /datadrive/ --ext img --print_model --n_val 10 --epochs 30 --save loss_tests/grad3 --loss 0.3*GradL1+0.7*L1
#python main.py --model EDSR --scale 4 --enable_rrl --half_resblocks --half_feats --reset --pre_train ../experiment/model/EDSR_baseline_x4.pt --dir_data /datadrive/ --ext img --print_model --n_val 10 --epochs 30 --save loss_tests/grad4 --loss 0.4*GradL1+0.6*L1
#python main.py --model EDSR --scale 4 --enable_rrl --half_resblocks --half_feats --reset --pre_train ../experiment/model/EDSR_baseline_x4.pt --dir_data /datadrive/ --ext bin --print_model --n_val 10 --epochs 30 --save loss_tests/grad5 --loss 0.5*GradL1+0.5*L1
#python main.py --model EDSR --scale 4 --enable_rrl --half_resblocks --half_feats --reset --pre_train ../experiment/model/EDSR_baseline_x4.pt --dir_data /datadrive/ --ext bin --print_model --n_val 10 --epochs 30 --save loss_tests/grad6 --loss 0.6*GradL1+0.4*L1
#python main.py --model EDSR --scale 4 --enable_rrl --half_resblocks --half_feats --reset --pre_train ../experiment/model/EDSR_baseline_x4.pt --dir_data /datadrive/ --ext bin --print_model --n_val 10 --epochs 30 --save loss_tests/grad7 --loss 0.7*GradL1+0.3*L1

#python main.py --model EDSR --scale 4 --enable_rrl --half_resblocks --half_feats --reset --pre_train ../experiment/model/EDSR_baseline_x4.pt --dir_data /datadrive/ --ext bin --print_model --n_val 10 --epochs 30 --save loss_tests/mae_VGG22 --loss 0.5*L1+0.5*VGG22
#python main.py --model EDSR --scale 4 --enable_rrl --half_resblocks --half_feats --reset --pre_train ../experiment/model/EDSR_baseline_x4.pt --dir_data /datadrive/ --ext bin --print_model --n_val 10 --epochs 30 --save loss_tests/mae_VGG54 --loss 0.5*L1+0.5*VGG54
#python main.py --model EDSR --scale 4 --enable_rrl --half_resblocks --half_feats --reset --pre_train ../experiment/model/EDSR_baseline_x4.pt --dir_data /datadrive/ --ext bin --print_model --n_val 10 --epochs 30 --save loss_tests/VGG22 --loss 1*VGG22
#python main.py --model EDSR --scale 4 --enable_rrl --half_resblocks --half_feats --reset --pre_train ../experiment/model/EDSR_baseline_x4.pt --dir_data /datadrive/ --ext bin --print_model --n_val 10 --epochs 30 --save loss_tests/VGG54 --loss 1*VGG54
