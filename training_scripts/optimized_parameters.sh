#!/bin/bash
#SBATCH --job-name="optimized_parameters"
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 # use 1 GPU
#SBATCH --cpus-per-gpu=1
#SBATCH -o logs/optimized_parameters.o%j
#SBATCH --mem-per-gpu=40G

source $PATCHATTACK_HOME/venv/patch-attack/bin/activate
cd $PATCHATTACK_HOME

# FlowNetC
python attack_patch_withDefense.py --net FlowNetC       --defense none --optimizer sgd    --lr 100  --change_of_variables True  --custom_weight_path models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_none --n_patches_eval 1
python attack_patch_withDefense.py --net FlowNetC       --defense lgs  --optimizer ifgsm  --lr 0.01 --change_of_variables True  --custom_weight_path models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_lgs --n_patches_eval 1
python attack_patch_withDefense.py --net FlowNetC       --defense ilp  --optimizer ifgsm  --lr 0.01 --change_of_variables True  --custom_weight_path models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_ilp --n_patches_eval 1

# FlowNetCRobust
python attack_patch_withDefense.py --net FlowNetCRobust --defense none --optimizer ifgsm  --lr 0.01 --change_of_variables False --custom_weight_path models/_pretrained_weights/RobustFlowNetC.pth        --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_none --n_patches_eval 1
python attack_patch_withDefense.py --net FlowNetCRobust --defense lgs  --optimizer ifgsm  --lr 1.0  --change_of_variables False --custom_weight_path models/_pretrained_weights/RobustFlowNetC.pth        --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_lgs --n_patches_eval 1
python attack_patch_withDefense.py --net FlowNetCRobust --defense ilp  --optimizer ifgsm  --lr 0.1  --change_of_variables True  --custom_weight_path models/_pretrained_weights/RobustFlowNetC.pth        --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_ilp --n_patches_eval 1

# PWCNet
python attack_patch_withDefense.py --net PWCNet         --defense none --optimizer ifgsm  --lr 0.1  --change_of_variables True  --custom_weight_path models/_pretrained_weights/pwc_net_chairs.pth.tar    --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_none --n_patches_eval 1
python attack_patch_withDefense.py --net PWCNet         --defense lgs  --optimizer ifgsm  --lr 1.0  --change_of_variables False --custom_weight_path models/_pretrained_weights/pwc_net_chairs.pth.tar    --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_lgs --n_patches_eval 1
python attack_patch_withDefense.py --net PWCNet         --defense ilp  --optimizer ifgsm  --lr 1.0  --change_of_variables False --custom_weight_path models/_pretrained_weights/pwc_net_chairs.pth.tar    --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_ilp --n_patches_eval 1

# SpyNet
python attack_patch_withDefense.py --net SpyNet         --defense none --optimizer ifgsm  --lr 0.1  --change_of_variables True                                                                            --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_none --n_patches_eval 1
python attack_patch_withDefense.py --net SpyNet         --defense lgs  --optimizer ifgsm  --lr 1.0  --change_of_variables False                                                                           --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_lgs --n_patches_eval 1
python attack_patch_withDefense.py --net SpyNet         --defense ilp  --optimizer ifgsm  --lr 1.0  --change_of_variables False                                                                           --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_ilp --n_patches_eval 1

# RAFT
python attack_patch_withDefense.py --net RAFT           --defense none --optimizer ifgsm  --lr 1.0  --change_of_variables True  --custom_weight_path models/_pretrained_weights/raft-kitti.pth            --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_none --n_patches_eval 1
python attack_patch_withDefense.py --net RAFT           --defense lgs  --optimizer sgd    --lr 100  --change_of_variables True  --custom_weight_path models/_pretrained_weights/raft-kitti.pth            --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_lgs --n_patches_eval 1
python attack_patch_withDefense.py --net RAFT           --defense ilp  --optimizer ifgsm  --lr 1.0  --change_of_variables False --custom_weight_path models/_pretrained_weights/raft-kitti.pth            --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_ilp --n_patches_eval 1

# GMA
python attack_patch_withDefense.py --net GMA            --defense none --optimizer ifgsm  --lr 0.01 --change_of_variables True  --custom_weight_path models/_pretrained_weights/gma-kitti.pth             --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_none --n_patches_eval 1
python attack_patch_withDefense.py --net GMA            --defense lgs  --optimizer ifgsm  --lr 1.0  --change_of_variables False --custom_weight_path models/_pretrained_weights/gma-kitti.pth             --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_lgs --n_patches_eval 1
python attack_patch_withDefense.py --net GMA            --defense ilp  --optimizer ifgsm  --lr 1.0  --change_of_variables False --custom_weight_path models/_pretrained_weights/gma-kitti.pth             --dataset KittiRaw --dataset_stage training --loss acs_target --n 2500 --alpha 1e-8 --save_frequency 100 --eval_after True --custom_experiment_name train_ilp --n_patches_eval 1
