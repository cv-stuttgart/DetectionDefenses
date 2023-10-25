#!/bin/bash

#SBATCH --job-name="eval_patches"
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 # use 1 GPU
#SBATCH --cpus-per-gpu=1
#SBATCH -o logs/eval_patches.o%j
#SBATCH --mem-per-gpu=30G ## GMA: 2.7jGB, FlowFormer 6,3GB
# export CUDA_VISIBLE_DEVICES=1

source $PATCHATTACK_HOME/venv/patch-attack/bin/activate
cd $PATCHATTACK_HOME



# model to evaluate. net is first argument, weights is second
net=FlowNetC
weights=models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar  #raft-kitti.pth # models/_pretrained_weights/pwc_net_chairs.pth.tar #FlowNet2-C_checkpoint.pth.tar  #raft-kitti.pth #

# eval best n patches (with the same parameter combinations)
# n=1 # currently head -n ${n} is commented out

# defense on which patches were trained
for defense in none lgs ilp; do

    # folder where patches are stored
    folder=results/${net}_${defense}_evals/best-patches/
    # folder=results/Manual_${defense}_evals/best-patches/
    # number of images to evaluate on
    N=200
    
    
    # fixed parameters
    exp_name=eval_${net}_${defense}
    # exp_name=eval_Manual_${net}_${defense}
    s=15
    t=0.15
    o=8
    k=16
    r=5
    
    # first n patches from folder
    for patch in $(ls ${folder}) #| head -n ${n}) # | grep png)
    do
        echo "patch: $patch"
        
        ## no defense
        python evaluate_patch_withDefense.py --dataset Kitti15 --dataset_stage training --net $net -w $weights --patch_name ${folder}/${patch} --defense none --custom_experiment_name ${exp_name}_none --n $N &
        sleep 5 # wait for experiment to be created
        
        # LGS
        python evaluate_patch_withDefense.py --dataset Kitti15 --dataset_stage training --net $net -w $weights --patch_name ${folder}/${patch} --defense lgs --s $s --t $t --custom_experiment_name ${exp_name}_lgs --n $N &
        sleep 5 # wait for experiment to be created
        
        # ILP
        python evaluate_patch_withDefense.py --dataset Kitti15 --dataset_stage training --net $net -w $weights --patch_name ${folder}/${patch} --defense ilp --r $r --s $s --t $t --k $k --o $o --custom_experiment_name ${exp_name}_ilp --n $N
        sleep 5 # wait for experiment to be created
        
    done
done
exit
