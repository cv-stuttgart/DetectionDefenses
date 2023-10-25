#!/bin/bash  

#SBATCH --job-name="defenseParams"  
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH -o logs/attack_log.o%j
# export CUDA_VISIBLE_DEVICES=1

source $PATCHATTACK_HOME/venv/patch-attack/bin/activate
cd $PATCHATTACK_HOME

patch_name=patches/FlowNetC_none.png

## LGS
# s,t
for s  in 0 3 6 9 12 15 18 21 24 27 30
do
    for t in 0 .04 .08 .12 .16 .2 .24 .28 .32 .36 .4
    do
        echo "Running with s = $s and t = $t"
        # use some best performing patch
        python evaluate_patch_withDefense.py --dataset Kitti15 --dataset_stage evaluation --net FlowNetC -w models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar --patch_name $patch_name --defense lgs --r 5 --s $s --t $t --k 16 --o 8 --custom_experiment_name defense_parameter_t_s_lgs --n 200 &
    done
    wait
done
wait
python training_scripts/eval_defense_params.py --mlruns_dir mlruns --experiment_name FlowNetC_PatchAttack-with-defense_cd_u_defense_parameter_t_s_lgs_eval --variables t,s

k,o
for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 
do
    for o in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    do
        if [ $k -lt $o ]
        then
            continue
        fi
        echo "Running with k = $k and o = $o"
        # use some best performing patch
        python evaluate_patch_withDefense.py --k $k --o $o --dataset Kitti15 --dataset_stage evaluation --net FlowNetC -w models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar --patch_name $patch_name --defense lgs --r 5 --s 15 --t .15 --custom_experiment_name defense_parameter_k_o_lgs --n 200 &
    done
    wait
done
wait
python training_scripts/eval_defense_params.py --mlruns_dir mlruns --experiment_name FlowNetC_PatchAttack-with-defense_cd_u_defense_parameter_k_o_lgs_eval --variables k,o

# ILP
s,t
for s in 0 3 6 9 12 15 18 21 24 27 30
do
    for t in 0 .04 .08 .12 .16 .2 .24 .28 .32 .36 .4
    do
        echo "Running with s = $s and t = $t"
        # use some best performing patch
        python evaluate_patch_withDefense.py --dataset Kitti15 --dataset_stage evaluation --net FlowNetC -w models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar --patch_name $patch_name --defense ilp --r 5 --s $s --t $t --k 16 --o 8 --custom_experiment_name defense_parameter_t_s_ilp --n 200 &
    done
    wait
done
wait
python training_scripts/eval_defense_params.py --mlruns_dir mlruns --experiment_name FlowNetC_PatchAttack-with-defense_cd_u_defense_parameter_t_s_ilp_eval --variables t,s

# k,o
for k  in  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
do
    for o in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    do
        if [ $k -lt $o ]
        then
            continue
        fi
        echo "Running with k = $k and o = $o"
        # use some best performing patch
        python evaluate_patch_withDefense.py --k $k --o $o --dataset Kitti15 --dataset_stage evaluation --net FlowNetC -w models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar --patch_name $patch_name --defense ilp --r 5 --s 15 --t .15 --custom_experiment_name defense_parameter_k_o_ilp --n 200 &
    done
    wait
done
wait
python training_scripts/eval_defense_params.py --mlruns_dir mlruns --experiment_name FlowNetC_PatchAttack-with-defense_cd_u_defense_parameter_k_o_ilp_eval --variables k,o

# r
for r in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    echo "Running with r = $r"
    # use some best performing patch
    python evaluate_patch_withDefense.py --r $r --dataset Kitti15 --dataset_stage evaluation --net FlowNetC -w models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar --patch_name $patch_name --defense ilp --s 15 --t .15 --k 16 --o 8 --custom_experiment_name defense_parameter_r --n 200 &
done
wait
python training_scripts/eval_defense_params.py --mlruns_dir mlruns --experiment_name FlowNetC_PatchAttack-with-defense_cd_u_defense_parameter_r_eval --variables r