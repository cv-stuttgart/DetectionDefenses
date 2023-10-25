#!/bin/bash
#SBATCH --job-name="AnyNet"
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 # use 1 GPU
#SBATCH --cpus-per-gpu=1
#SBATCH -o logs/AnyNet.o%j
#SBATCH --mem-per-gpu=15G
# remark on Memory: FlowNetC:15, RAFT:35, GMA: 6.1GB per Net, PWCNet: 3.3GB per Net, flowformer 30GB per Net, robustflownetc 2.9GB per Net


source $PATCHATTACK_HOME/venv/patch-attack/bin/activate
cd $PATCHATTACK_HOME

# variables to change for different models
net=FlowNetC # RAFT FlowNetCRobust SpyNet PWCNet GMA FlowFormer
weights=models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar

# defenses
for defense in none lgs ilp; do
    loss=acs_${defense}
    # if defense is none, use acs_target
    if [ $defense == "none" ]; then
        loss=acs_target
    fi
    
    # fixed parameters
    s=15
    t=0.15
    o=8
    k=16
    r=5
    exp_name=train_${defense}
    
    opt=sgd
    for lr in 10 100 ; do
        for cov in True False; do
            # first seed handeled separately, so the mlrun experiment is created and the other seeds do not interfere with creating the experiment (they also try to create it and fail because it exists)
            python attack_patch_withDefense.py --patch_size 100 --net $net --custom_weight_path $weights --dataset KittiRaw --dataset_stage training --shuffle True --defense $defense --k $k --o $o --t $t --s $s --r $r --optimizer $opt --lr $lr --gamma 1 --loss $loss --target neg_flow --n 2500 --alpha 1e-8 --save_frequency 1000 --change_of_variables $cov --batch_size 1 --steps 1 --eval_after True --custom_experiment_name $exp_name --seed 2 --scheduler exponential-lr --n_patches_eval 1 &
            # wait for 10 seconds to make sure the experiment is created
            sleep 10
            for seed in 4 8 16; do # 42
                python attack_patch_withDefense.py --patch_size 100 --net $net --custom_weight_path $weights --dataset KittiRaw --dataset_stage training --shuffle True --defense $defense --k $k --o $o --t $t --s $s --r $r --optimizer $opt --lr $lr --gamma 1 --loss $loss --target neg_flow --n 2500 --alpha 1e-8 --save_frequency 1000 --change_of_variables $cov --batch_size 1 --steps 1 --eval_after True --custom_experiment_name $exp_name --seed $seed --scheduler exponential-lr --n_patches_eval 1 &
            done
            wait
        done
    done
    opt=ifgsm
    for lr in .01 .1 1; do 
        for cov in True False; do
            for seed in 2 4 8 16; do # 42
                python attack_patch_withDefense.py --patch_size 100 --net $net --custom_weight_path $weights --dataset KittiRaw --dataset_stage training --shuffle True --defense $defense --k $k --o $o --t $t --s $s --r $r --optimizer $opt --lr $lr --gamma 1 --loss $loss --target neg_flow --n 2500 --alpha 1e-8 --save_frequency 1000 --change_of_variables $cov --batch_size 1 --steps 1 --eval_after True --custom_experiment_name $exp_name --seed $seed --scheduler exponential-lr --n_patches_eval 1 &
            done
            wait
        done
    done
done


# one example for GMA
# python attack_patch_withDefense.py --patch_size 100 --net GMA --custom_weight_path models/_pretrained_weights/gma-kitti.pth --dataset KittiRaw --dataset_stage training --shuffle True --defense none --k 16 --o 8 --t 0.15 --s 15 --r 5 --optimizer sgd --lr 10 --gamma 1 --loss acs_target --target neg_flow --n 2500 --alpha 1e-8 --save_frequency 1000 --change_of_variables True --batch_size 1 --steps 1 --eval_after True --custom_experiment_name train_none --seed 2 --scheduler exponential-lr --n_patches_eval 5

# session on gpu 3
# srun --gres=gpu:1 --cpus-per-gpu=1 --mem-per-gpu=30G --pty /bin/bash
