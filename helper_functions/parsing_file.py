import argparse
import json

def create_parser(stage=None, attack_type=None):
    stage = stage.lower()
    attack_type = attack_type.lower()
    if stage not in ['training', 'evaluation', 'submission']:
        raise ValueError('To create a parser the stage has to be specified. Please choose one of "training" or "evaluation"')
    if attack_type not in ["patch_attack", "fgsm", "pcfa", "no_attack","patch_attack_withdefense"]:
            raise ValueError('To create a parser the attack type has to be specified. Please choose one of "pcfa", "fgsm" or "patch_attack"]:')


    parser = argparse.ArgumentParser(usage='%(prog)s [options (see below)]')

    # network arguments
    net_args = parser.add_argument_group(title='network arguments')
    net_args.add_argument('--net', default='RAFT', choices=['RAFT', 'GMA', 'FlowFormer', 'PWCNet', 'SpyNet', 'FlowNet2', 'FlowNet2C', 'FlowNetC', 'FlowNet2S', 'FlowNetCRobust'],
        help="specify the network under attack")
    net_args.add_argument('-w','--custom_weight_path', default='',help="specify path to load weights from. By default loads to the given net default weights from `models/_pretrained_weights`. Does not work with SpyNet")

    # Dataset arguments
    dataset_args = parser.add_argument_group(title="dataset arguments")
    dataset_args.add_argument('--dataset', default='KittiRaw', choices=['Kitti15', 'Sintel', 'Spring', 'SintelSplitZhao', 'SintelSplitYang', 'SpringSplitScheurer', 'HD1KSplitScheurer', 'DrivingSample', 'KittiRaw'],
        help="specify the dataset which should be used for evaluation")
    dataset_args.add_argument('--dataset_stage', default='training', choices=['training', 'evaluation'],
        help="specify the dataset stage ('training' or 'evaluation') that should be used.")
    dataset_args.add_argument('--small_run', action='store_true',
        help="for testing purposes: if specified the dataloader will on load 32 images")

    # Sintel specific:
    sintel_args = parser.add_argument_group(title="sintel specific arguments")
    sintel_args.add_argument('--dstype', default='final', choices=['clean', 'final'],
        help="[only sintel] specify the dataset type for the sintel dataset")

    # Data saving
    data_save_args = parser.add_argument_group(title="data saving arguments")
    data_save_args.add_argument('--output_folder', default='experiment_data',
        help="data that is logged during training and evaluation will be saved there")
    data_save_args.add_argument('--small_save', action='store_true',
        help="if specified potential extended output will only be produced for the first 32 images.")
    data_save_args.add_argument('--save_frequency', type=int, default=1,
            help="specifies after how many batches intermediate results (patch, input images, flows) should be saved. Default: 1 (save after every batch/image). If --no_save is specified, this overwrites any save_frequency.")
    data_save_args.add_argument('--no_save', action='store_true',
        help="if specified no extended output (like distortions/patches) will be written. This overwrites any value specified by save_frequency.")
    data_save_args.add_argument('--unregistered_artifacts', action='store_true', default=False,
        help="if this flag is used, artifacts are saved to the output folder but not registered. This might save time during training.")
    data_save_args.add_argument('-expname','--custom_experiment_name', default='',
        help="specify a custom mlflow experiment name. The given string is concatenated with the automatically generated one.")

    # Target setup
    target_args = parser.add_argument_group(title="target flow arguments")
    target_args.add_argument('--target', default='neg_flow', choices=['zero', 'neg_flow', 'custom', 'scaled_random_flow', 'scaled_constant_flow'],
        help="specify the attack target as one flow type out of 'zero', 'neg_flow', 'custom','scaled_random_flow'and 'scaled_constant_flow'. Additionally provide a '--custom_target_path' if 'custom' is chosen")
    target_args.add_argument('--custom_target_path', default='',
        help="specify path to a custom target flow")
    target_args.add_argument('--flow_target_scale', default=1,
        help="[scaled_random_flow and scaled_constant_flow target only] A scaling factor by which the flow is multiplied.")

    if attack_type in ["patch_attack_withdefense"]:
        patch_att_def_args = parser.add_argument_group(title="patch attack with defense arguments")

        patch_att_def_args.add_argument("--defense",dest="defense",default="none",help="Define defense", choices=['none','lgs','ilp'])
        patch_att_def_args.add_argument("--adversary",dest="adversary",default="patch-adversary",help="Choose Adversarial Model", choices=['patch-adversary','ReparametrizedAdversarialPatch'])
        patch_att_def_args.add_argument("--patch_size",dest="patch_size",type=int,default=50,help="size of the patch")
        patch_att_def_args.add_argument("--loss",help="overwrites loss argument. Only for patchattack with defense. For Adam need to use change of variables. acs_none uses the acs to the target", choices=["acs","acs_target","acs_none","acs_lgs","acs_ilp","aee_lgs","aee_ilp","aee_target","mse","mse_lgs","mse_ilp"])
        patch_att_def_args.add_argument("--k",dest="k",type=int,default=16,help="blocksize")
        patch_att_def_args.add_argument("--o",dest="o",type=int,default=8,help="overlap")
        patch_att_def_args.add_argument("--t",dest="t",type=float,default=0.15,help="blockwise filtering threshold")
        patch_att_def_args.add_argument("--s",dest="s",type=float,default=15.0,help="smoothing/scaling parameter depending on the defense")
        patch_att_def_args.add_argument("--r",dest="r",type=int,default=5,help="inpainting radius")
        patch_att_def_args.add_argument("--optimizer",dest="optimizer",default="ifgsm",help="optimizer. LBFGS doesn't work without change_of_variables",choices=["clipped-pgd","ifgsm","adam","lbfgs", "sgd"])
        patch_att_def_args.add_argument("--scheduler",dest="scheduler",default="exponential-lr",help="scheduler", choices=["exponential-lr","OneCycleLR"])
        patch_att_def_args.add_argument("--gamma",dest="gamma",type=float,default=1.0,help="additional parameter for the scheduler. For exponential-lr: gamma is the decay rate. For OneCycleLR: gamma is ptc_start, the percentage of the cycle spent increasing the learning rate")
        patch_att_def_args.add_argument("--alpha",dest="alpha",type=float,default=0.001,help="weigth of additional error term")
        patch_att_def_args.add_argument("--n",dest="n",type=int,default=1000,help="number of iterations. Set n=-1 to evaluate all images in the dataset")
        patch_att_def_args.add_argument("--lr",dest="lr",type=float,default=.1,help="learning rate of the optimizer")
        patch_att_def_args.add_argument("--max_delta",dest="max_delta",type=float,default=0.008,help="maximum delta for parameter update. Gradients will be clipped when using Projected Gradient Descent")
        patch_att_def_args.add_argument("--batch_size",dest="batch_size",type=int,default=1,help="Batch size for training the patch.")
        patch_att_def_args.add_argument("--steps",dest="steps",type=int,default=1,help="number of steps per batch. If larger than one, multiple updates are performed per patch.")
        patch_att_def_args.add_argument("--seed",dest="seed",type=int,default=-1,help="random seed. If -1, a random seed is generated")
        patch_att_def_args.add_argument("--change_of_variables",type=lambda x: x.lower()=='true',default=False,help="If 'true' a change of variable is performed.")
        patch_att_def_args.add_argument("--eval_after",type=lambda x: x.lower()=='true',default=False,help="If 'true' the patch is evaluated after the training. This starts a new mlflow experiment by running the evaluation script on kitti15.")
        patch_att_def_args.add_argument("--shuffle",type=lambda x: x.lower()=='true',default=True, help="If 'true' the dataset is shuffled")
        patch_att_def_args.add_argument("--crop_shape", nargs='+', default=[], help="Crop shape. If empty, no cropping is performed. Usage: `--crop_shape 320 1216`")
        patch_att_def_args.add_argument("--n_patches_eval", type=int, default=1, help="Amount of patches to evaluate during the end of the training. This is to see if the training is still changing during the last iterations.")
        patch_att_def_args.add_argument("--patch_name",dest="patch_name",type=str,default='',help="For evaluatioin, give the full path to the patch")
        patch_att_def_args.add_argument("--run_id",dest="run_id",type=str,default='',help="For eavaluation: Mlflow run id. Loads the following parameters from mlflow: k, o, t, s, r. The parameters patch_name, custom_weight_path are loaded if not specified. The patch is the last saved patch of the specified run.")
    return parser
