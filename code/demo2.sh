#LOCAL SCRIPTS (for testing)

#test training-time 
#test normal dataloader training time 
#python main.py --model EDSR --scale 4 --save EDSR_baseline_x2 --reset --cpu --dir_data ~/Documents/ --n_train 3 --chop --n_val 1 --offset_val 800 --batch_size 1 --test_every 1

#test rrl dataloader training time
#python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --cpu --dir_data ~/Documents --ext img --n_val 1 --offset_val 800 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --n_threads 1 --batch_size 1 --test_every 1

#test test-time
#test normal dataloader test time
#python main.py --data_test DIV2K --cpu --dir_data ~/Documents/ --ext img --n_val 1 --offset_val 800 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only

#test rrl dataloader test time
#python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --cpu --dir_data ~/Documents --ext img --n_val 1 --offset_val 800 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --save_results --n_threads 1


#DEV DONE; EXPERIMENTS SCRIPTS
#   Baseline EDSR scripts (testing if halving resblocks is better or halving features is better)
#       x4 scale, branch = 1 
#       python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --ext img --scale 4 --n_channel_in 64 --model EDSR --branch_num 1 --half_resblocks --n_resblocks 16 --n_feats 64 --pre_train ../experiment/model/EDSR_baseline_x4.pt --epochs 30 --print_model --dir_data /datadrive --cpu
#       python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --ext img --scale 4 --n_channel_in 64 --model EDSR --branch_num 1 --half_feats --n_resblocks 16 --n_feats 64 --pre_train ../experiment/model/EDSR_baseline_x4.pt --epochs 30 --print_model --dir_data /datadrive --save test1
#       python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --ext img --scale 4 --n_channel_in 64 --model EDSR --branch_num 1 --half_resblocks --half_feats --n_resblocks 16 --n_feats 64 --pre_train ../experiment/model/EDSR_baseline_x4.pt --epochs 30 --print_model --dir_data /datadrive --save test4

#       x2 scale, branch = 1 
#       python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --ext img --scale 2 --n_channel_in 64 --model EDSR --branch_num 1 --half_resblocks --n_resblocks 16 --n_feats 64 --pre_train ../experiment/model/EDSR_baseline_x2.pt --epochs 30 --print_model --dir_data /datadrive --reset 
#       python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --ext img --scale 2 --n_channel_in 64 --model EDSR --branch_num 1 --half_feats --n_resblocks 16 --n_feats 64 --pre_train ../experiment/model/EDSR_baseline_x2.pt --epochs 30 --print_model --dir_data /datadrive --save test2
#       python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --ext img --scale 2 --n_channel_in 64 --model EDSR --branch_num 1 --half_resblocks --half_feats --n_resblocks 16 --n_feats 64 --pre_train ../experiment/model/EDSR_baseline_x2.pt --epochs 30 --print_model --dir_data /datadrive --save test5

#       x3 scale, branch = 1
#       python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --ext img --scale 3 --n_channel_in 64 --model EDSR --branch_num 1 --half_resblocks --n_resblocks 16 --n_feats 64 --pre_train ../experiment/model/EDSR_baseline_x3.pt --epochs 30 --print_model --dir_data /datadrive --reset
#       python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --ext img --scale 3 --n_channel_in 64 --model EDSR --branch_num 1 --half_feats --n_resblocks 16 --n_feats 64 --pre_train ../experiment/model/EDSR_baseline_x3.pt --epochs 30 --print_model --dir_data /datadrive --save test3
#       python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --ext img --scale 3 --n_channel_in 64 --model EDSR --branch_num 1 --half_resblocks --half_feats --n_resblocks 16 --n_feats 64 --pre_train ../experiment/model/EDSR_baseline_x3.pt --epochs 30 --print_model --dir_data /datadrive --save test6

#   Final EDSR scripts
#       #x4 scale, branch = 1 
#       python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --ext img --scale 4 --n_channel_in 64 --chop --model EDSR --branch_num 1 --half_resblocks --n_resblocks 32 --n_feats 256 --pre_train ../experiment/model/EDSR_x4.pt --print_model --dir_data /datadrive --reset --cpu

#       #x2 scale, branch = 1 
#       python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --ext img --scale 2 --n_channel_in 64 --chop --model EDSR --branch_num 1 --half_resblocks --n_resblocks 32 --n_feats 256 --pre_train ../experiment/model/EDSR_x2.pt --print_model --dir_data /datadrive --reset --cpu

#       #x2 scale, branch = 1 
#       python main.py --data_train rrl --rrl_data DIV2K --data_test rrl --ext img --scale 3 --n_channel_in 64 --chop --model EDSR --branch_num 1 --half_resblocks --n_resblocks 32 --n_feats 256 --pre_train ../experiment/model/EDSR_x3.pt --print_model --dir_data /datadrive --reset --cpu

