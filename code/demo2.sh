#LOCAL SCRIPTS (for testing)

#test training-time 
#python main.py --model EDSR --scale 4 --save EDSR_baseline_x2 --reset --cpu --dir_data ~/Documents/ --n_train 3 --chop --n_val 1 --offset_val 800 --batch_size 1 --test_every 1

#test test-time
#python main.py --data_test DIV2K --cpu --dir_data ~/Documents/ --ext img --n_val 1 --offset_val 800 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only