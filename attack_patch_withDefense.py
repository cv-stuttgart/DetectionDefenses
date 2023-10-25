# Code is influenced by the flowattack main.py file by Ranjan et al.
# https://github.com/anuragranj/flowattack

#%% Standard libraries
import sys
import os.path as op
import torch
# torch.set_num_threads(4)
import torchvision
import numpy as np
import mlflow
import matplotlib.pyplot as plt

from mlflow import log_metric,log_param,log_artifact
from tqdm import tqdm

# Custom libraries
from helper_functions import ownutilities,parsing_file, logging, targets
from helper_functions.config_specs import Conf
from helper_functions.defenses import LGS,ILP, Joint2ndGradMag, JointGradMag
from helper_functions.custom_optimizer import ClippedPGD,IFGSM
from helper_functions.patch_adversary import PatchAdversary
from helper_functions.losses import aee_masked, acs_masked,aae_masked,mse_masked
from helper_functions.ownutilities import show_images # debugging

   
#%% loss functions for the patch attack
def acs(A,D,F_attacked,F_unattacked,M,I1_attacked,I2_attacked,I1_unattacked,I2_unattacked, target):
    return acs_masked(F_unattacked,F_attacked,1-M)
def acs_target(A,D,F_attacked,F_unattacked,M,I1_attacked,I2_attacked,I1_unattacked,I2_unattacked, target):
    return -acs_masked(target,F_attacked,1-M)
def acs_lgs(A,D,F_attacked,F_unattacked,M,I1_attacked,\
                    I2_attacked,I1_unattacked,I2_unattacked,target):
    """Similar to acs_target but with the additional LGS term"""
    return -acs_masked(target,F_attacked,1-M)+ args.alpha*JointGradMag()(A.M*A.get_P(),"forward").sum()/A.M.sum()
def acs_ilp(A,D,F_attacked,F_unattacked,M,I1_attacked,\
                    I2_attacked,I1_unattacked,I2_unattacked,target):
    """Similar to acs_target but with the additional ILP term"""
    # return acs_masked(F_unattacked,F_attacked,1-M)+ args.alpha*Joint2ndGradMag()(A.M*A.get_P(),"forward").sum()/A.M.sum()
    return -acs_masked(target,F_attacked,1-M)+ args.alpha*Joint2ndGradMag()(A.M*A.get_P(),"forward").sum()/A.M.sum()

def aee_target(A,D,F_attacked,F_unattacked,M,I1_attacked,I2_attacked,\
                    I1_unattacked,I2_unattacked,target):
    return aee_masked(target,F_attacked,1-M)
def aee_lgs(A,D,F_attacked,F_unattacked,M,I1_attacked,\
                    I2_attacked,I1_unattacked,I2_unattacked,target):
    return aee_masked(target,F_attacked,1-M)+ args.alpha*JointGradMag()(A.M*A.get_P(),"forward").sum()/A.M.sum()
def aee_ilp(A,D,F_attacked,F_unattacked,M,I1_attacked,\
                    I2_attacked,I1_unattacked,I2_unattacked,target):
    return aee_masked(target,F_attacked,1-M)+ args.alpha*Joint2ndGradMag()(A.M*A.get_P(),"forward").sum()/A.M.sum()

def mse_target(A,D,F_attacked,F_unattacked,M,I1_attacked,I2_attacked,\
                    I1_unattacked,I2_unattacked,target):
    return mse_masked(target,F_attacked,1-M)
def mse_lgs(A,D,F_attacked,F_unattacked,M,I1_attacked,\
                    I2_attacked,I1_unattacked,I2_unattacked,target):
    return mse_masked(target,F_attacked,1-M)+ args.alpha*JointGradMag()(A.M*A.get_P(),"forward").sum()/A.M.sum()
def mse_ilp(A,D,F_attacked,F_unattacked,M,I1_attacked,\
                    I2_attacked,I1_unattacked,I2_unattacked,target):
    return mse_masked(target,F_attacked,1-M)+ args.alpha*Joint2ndGradMag()(A.M*A.get_P(),"forward").sum()/A.M.sum()

#%% Logger being called during the attack
class CustomLogger:
    def __init__(self, n, save_frequency, output_folder, unregistered_artifacts):
        self.n = n
        self.save_frequency = save_frequency
        self.output_folder = output_folder
        self.unregistered_artifacts = unregistered_artifacts

        self.i = 0

        self.sum_aee_def_advdef = 0
        self.sum_aee_tgt_advdef = 0
        self.sum_aee_def_tgt = 0

        self.sum_acs_adv_advdef = 0
        self.sum_acs_tgt_advdef = 0
        self.sum_acs_def_tgt = 0

        self.sum_mse_adv_advdef = 0
        self.sum_mse_tgt_advdef = 0
        self.sum_mse_def_tgt = 0

        self.sum_patch_gradmag = 0
        self.sum_patch_2ndgradmag = 0

    @torch.no_grad()
    def update(self, I1, I2, I1_p, I2_p, I1_attacked_def, I2_attacked_def, I1_unattacked_def, I2_unattacked_def, A, F_attacked_def, F_unattacked_def, target, M, flow, has_gt):
        """ Update the logger with the current iteration's results. (before opt.step!) """
        aee_def_advdef = aee_masked(F_attacked_def, F_unattacked_def, 1-M)
        aee_tgt_advdef = aee_masked(target,F_attacked_def,1-M)
        aee_def_tgt = aee_masked(F_unattacked_def, target)

        acs_adv_advdef = acs_masked(F_attacked_def, F_unattacked_def, 1-M)
        acs_tgt_advdef = acs_masked(target,F_attacked_def,1-M)
        acs_def_tgt = acs_masked(F_unattacked_def, target)

        mse_adv_advdef = mse_masked(F_attacked_def, F_unattacked_def, 1-M)
        mse_tgt_advdef = mse_masked(target,F_attacked_def,1-M)
        mse_def_tgt = mse_masked(F_unattacked_def, target)

        patch_gradmag = JointGradMag()(A.M*A.get_P(),"forward").sum()
        patch_2ndgradmag = Joint2ndGradMag()(A.M*A.get_P(),"forward").sum()

        self.sum_aee_def_advdef += aee_def_advdef
        self.sum_aee_tgt_advdef += aee_tgt_advdef
        self.sum_aee_def_tgt += aee_def_tgt

        self.sum_acs_adv_advdef += acs_adv_advdef
        self.sum_acs_tgt_advdef += acs_tgt_advdef
        self.sum_acs_def_tgt += acs_def_tgt

        self.sum_mse_adv_advdef += mse_adv_advdef
        self.sum_mse_tgt_advdef += mse_tgt_advdef
        self.sum_mse_def_tgt += mse_def_tgt

        self.sum_patch_gradmag += patch_gradmag
        self.sum_patch_2ndgradmag += patch_2ndgradmag

        logging.log_metrics(self.i,
                            ("aee_def_advdef",aee_def_advdef),
                            ("aee_tgt_advdef",aee_tgt_advdef),
                            ("aee_def_tgt",aee_def_tgt),
                            ("acs_adv_advdef",acs_adv_advdef),
                            ("acs_tgt_advdef",acs_tgt_advdef),
                            ("acs_def_tgt",acs_def_tgt),
                            ("mse_adv_advdef",mse_adv_advdef),
                            ("mse_tgt_advdef",mse_tgt_advdef),
                            ("mse_def_tgt",mse_def_tgt),
                            ("patch_gradmag",patch_gradmag),
                            ("patch_2ndgradmag",patch_2ndgradmag))
        
    
        # only save every save_frequency iterations, the last iteration, and every 15 iterations after args.n
        if not (self.i==self.n-1 or self.i%self.save_frequency == 0 or (self.i>self.n and (self.i-self.n)%15==14)):
            self.i += 1
            return

        logging.save_tensor(A.get_P(Mask=True), f"Patch",self.i,self.output_folder,self.unregistered_artifacts)
        logging.save_tensor(A.P, f"Untransformed_Patch",self.i,self.output_folder,self.unregistered_artifacts)
        logging.save_image(A.get_P(Mask=True),self.i,self.output_folder,image_name="Patch",unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
        logging.save_image(A.get_P(),self.i,self.output_folder,image_name="Patch_no_mask",unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
        if not args.no_save:
            logging.save_image(I1_attacked_def, self.i, self.output_folder, image_name="I1_attacked_def", unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_image(I2_attacked_def, self.i, self.output_folder, image_name="I2_attacked_def", unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_image(I1_unattacked_def, self.i, self.output_folder, image_name="I1_unattacked_def", unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_image(I2_unattacked_def, self.i, self.output_folder, image_name="I2_unattacked_def", unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_image(I1_p, self.i, self.output_folder, image_name="I1_attacked", unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_image(I2_p, self.i, self.output_folder, image_name="I2_attacked", unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
            log_metric("last_saved", self.i)

            max_flow_gt = 0
            if has_gt.all():
                max_flow_gt = ownutilities.maximum_flow(flow)
            max_flow = np.max([max_flow_gt, 
                            ownutilities.maximum_flow(F_unattacked_def), 
                            ownutilities.maximum_flow(F_attacked_def)])

            logging.save_flow(F_attacked_def, self.i, self.output_folder, flow_name='flow_pred_best', auto_scale=False, max_scale=max_flow, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_flow(F_unattacked_def, self.i, self.output_folder, flow_name='flow_pred_init', auto_scale=False, max_scale=max_flow, unregistered_artifacts=self.unregistered_artifacts)
        self.i += 1
    
#%% Main training loop function
def train(A,N,D,loss,dl,optimizer,scheduler,args,device,seed=None):
    """Training procedure with Attack, Net, Defense, Data ...

    Args:
        A (_type_): Adversary
        N (_type_): Network
        D (_type_): Defense
        loss (_type_): Loss function
        dl (_type_): Data loader is iterable
        optimizer (_type_): Optimizer
        scheduler (_type_): Scheduler
        n (_type_): Number of steps
        name (_type_): Name of the patch
        args (_type_): Arguments from the command line
        reproducible (bool, optional): Whether to set a seed or not. Defaults to True.

    Returns:
        A,data: Trained adversary and optimization data
    """
    n = args.n
    print(f"Training for {n} iterations with {args.steps} steps per iteration")
    unit_images = ownutilities.model_takes_unit_input(args.net)
    
    if seed is not None:
        import random
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False # this being false may reduce perfomance (https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking)
        torch.backends.cudnn.deterministic=True
    
    if args.crop_shape: # is not empty
        Cropper = torchvision.transforms.RandomCrop([int(x) for x in args.crop_shape])
    else:
        Cropper = lambda x: x

    logger = CustomLogger(args.n, args.save_frequency, args.output_folder, args.unregistered_artifacts)
    for i, (I1_orig,I2_orig,flow,has_gt) in enumerate(tqdm(dl, total=n+max((args.n_patches_eval-1)*15,0))):
        if i == n+max((args.n_patches_eval-1)*15,0):
            break
        I1 = I1_orig.clone().to(device)
        I2 = I2_orig.clone().to(device)

        # 1.) Preprocess images
        if not unit_images:
            # If the model takes unit input, ownutilities.preprocess_img will transform images into [0,1].
            # Otherwise, do transformation here
            I1 = I1/255.
            I2 = I2/255.

        I1,I2 = Cropper(torch.cat([I1,I2],dim=0)).chunk(2,dim=0)
        padder, [I1, I2] = ownutilities.preprocess_img(args.net, I1, I2)


        def closure(I1 = I1, I2 = I2, flow = flow, has_gt = has_gt, logger = None):
            # 2.) Apply Attack
            A.zero_grad()
            I1_p,I2_p,M,y,x = A(I1,I2)#,y = 150,x=600)
            [M] = padder.unpad(M)

            # 3.) Apply Defense 
            if isinstance(D,LGS) or isinstance(D,ILP):
                I1_attacked_def,I2_attacked_def = D(I1_p,I2_p,M) 
                I1_unattacked_def,I2_unattacked_def = D(I1,I2,M)
            else:
                I1_attacked_def,I2_attacked_def = I1_p,I2_p
                I1_unattacked_def,I2_unattacked_def = I1,I2
                
            # 4.) Predict
            F_attacked_def = ownutilities.compute_flow(N,"scaled_input_model",I1_attacked_def,I2_attacked_def)
            [F_attacked_def] = ownutilities.postprocess_flow(args.net, padder, F_attacked_def)
            F_unattacked_def = ownutilities.compute_flow(N,"scaled_input_model",I1_unattacked_def,I2_unattacked_def)
            [F_unattacked_def] = ownutilities.postprocess_flow(args.net, padder, F_unattacked_def)

            # 4.5) get Target
            target = targets.get_target(args.target, F_unattacked_def, flow_target_scale=args.flow_target_scale, custom_target_path=args.custom_target_path, device=device)

            # 5.) Loss Computation
            l=loss(A,D,F_attacked_def,F_unattacked_def,M,I1_attacked_def,I2_attacked_def,I1_unattacked_def,I2_unattacked_def, target)
            l.backward()

            if logger: # dont update logger if using internal steps of optimizers
                logger.update(I1, I2, I1_p, I2_p, I1_attacked_def, I2_attacked_def, I1_unattacked_def, I2_unattacked_def, A, F_attacked_def, F_unattacked_def, target, M, flow, has_gt)
            
            return l

        for j in range(args.steps): # multiple steps per iteration
            # 6.) Update
            l = closure(logger = logger)
            # break if loss is nan
            if torch.isnan(l) or torch.isnan(A.P).any():
                print("Loss is nan, stopping training")
                raise ValueError("Loss is nan, stopping training")
            
            if args.optimizer == "lbfgs":
                # for LBFGS the above call of closure is unnecessary but it is needed for the other optimizers and for logging
                optimizer.step(closure) 
            else:
                optimizer.step()
            
            # 7.) clip to unit interval
            if not args.change_of_variables and args.optimizer not in ["ifgsm","pgd"]:
                with torch.no_grad():
                    A.P.clamp_(0,1)
            # 8.) logging
            logging.log_metrics(i*args.steps+j,("loss",l.item()))
        
        if i<args.n: # only update scheduler if not in evaluation phase
            scheduler.step()

    return A

#%% Function to set up the attack and mlflow
def train_patch(args):
    """Training procedure for a patch. This function loads the data, the network, the adversary, the defense and the loss function. It then calls the train function.

    Args:
        args (_type_): Arguments from the command line
    """
    experiment_id, folder_path, folder_name = logging.mlflow_experimental_setup(args.output_folder, args.net, "PatchAttack-with-defense", True, True, args.custom_experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=folder_name) as run:

        ## MLflow logging
        print("\nStarting Defended Patch Attack:")
        print()
        print("\tModel:                   %s" % (args.net))
        print("\tDataset:                 %s" % (args.dataset))
        print("\tDefense:                 %s" % (args.defense))
        print("\tLoss:                    %s" % (args.loss))
        print()
        print("\tTarget:                  %s" % (args.target))
        print("\tOptimizer:               %s" % (args.optimizer))
        print("\tOptimizer steps:         %d" % (args.n))
        print("\tOptimizer LR:            %f" % (args.lr))
        print()
        print("\tk:                       %d" % (args.k))
        print("\to:                       %d" % (args.o))
        print("\tt:                       %d" % (args.t))
        print("\ts:                       %d" % (args.s))
        print("\tr:                       %d" % (args.r))
        print()
        print("\tOutputfolder:            %s" % (folder_path))
        print("\tMlflow experiment id:    %s" % (experiment_id))
        print("\tMlflow run id:           %s" % (run.info.run_id))
        print()
        
        try:
            log_param("full_command", "python "+" ".join(sys.argv))
        except mlflow.exceptions.MlflowException:
            print("full_command too long for mlflow")
        log_param("outputfolder", folder_path)
        # distortion_folder_name = "patches"
        # distortion_folder_path = folder_path
        # distortion_folder = logging.create_subfolder(distortion_folder_path, distortion_folder_name)

        model_takes_unit_input = ownutilities.model_takes_unit_input(args.net)
        logging.log_model_params(args.net,model_takes_unit_input)
        logging.log_dataset_params(args.dataset, 1, 1, args.dstype, args.dataset_stage)
        logging.log_attack_params("PatchAttack-with-defense", None, args.target, True, True, random_scale=args.flow_target_scale, custom_target_path=args.custom_target_path)
        log_param("patch_size", args.patch_size)
        log_param("optimizer", args.optimizer)
        log_param("loss",args.loss)
        log_param("lr", args.lr)
        log_param("flow_target_scale", args.flow_target_scale)
        log_param("custom_target_path", args.custom_target_path)
        log_param("scheduler", args.scheduler)
        log_param("gamma", args.gamma)
        log_param("defense", args.defense)
        log_param("k", args.k)
        log_param("o", args.o)
        log_param("t", args.t)
        log_param("s", args.s)
        log_param("r", args.r)
        log_param("n", args.n)
        log_param("alpha", args.alpha)
        log_param("max_delta", args.max_delta)
        log_param("change_of_variables", args.change_of_variables)
        log_param("crop_shape", args.crop_shape)
        log_param("n_patches_eval", args.n_patches_eval)
        log_param("save_frequency", args.save_frequency)
        log_param("no_save", args.no_save)
        if not args.eval_after:
            args.__dict__["n_patches_eval"] = 0
        

        if Conf.config('useCPU') or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        print(f"Setting Device to {device}")

        ## Start experiment
        # model
        print(f"Loading model {args.net}...")
        model, path_weights = ownutilities.import_and_load(args.net, custom_weight_path=args.custom_weight_path, make_unit_input=not model_takes_unit_input, variable_change=False, make_scaled_input_model=True,device=device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        print("Done\n")
        log_param("model_path_weights", args.custom_weight_path)
            
        # defense
        if args.defense == "lgs":
            D = LGS(args.k,args.o,args.t,args.s,"forward")
        elif args.defense == "ilp":
            D = ILP(args.k,args.o,args.t,args.s,args.r,"forward")
        elif args.defense == "none":
            D = None
        else:
            print("invalid defense name")
            return
        
        
        # adversary
        A = PatchAdversary(None,size=args.patch_size,angle=[-10,10],scale=[0.95,1.05], change_of_variable=args.change_of_variables).to(device)

        # dataset 
        print(f"Preparing data from {args.dataset} {args.dataset_stage}\n...", end=" ")
        data_loader, has_gt = ownutilities.prepare_dataloader(args.dataset_stage, 
                                                        dataset=args.dataset,
                                                        shuffle=args.shuffle,
                                                        batch_size=args.batch_size,
                                                        #   small_run=args.small_run, TODO
                                                        dstype=args.dstype, 
                                                        num_repeats=4
                                                        )
        print("Done\n")
        
        # optimizer
        if args.optimizer == "clipped-pgd":
            O = ClippedPGD(A.parameters(),lr=args.lr,min_=0,max_=1,max_delta=args.max_delta)
        elif args.optimizer == "ifgsm":
            if args.change_of_variables:
                O = IFGSM(A.parameters(),lr=args.lr,min_=-100,max_=100) # if change of variables is used, the range of the patch is [-100,100] and not [0,1] because the patch is scaled to [0,1] before the forward pass
            else:
                O = IFGSM(A.parameters(),lr=args.lr,min_=0,max_=1)
        elif args.optimizer == "lbfgs":
            O = torch.optim.LBFGS(A.parameters(),lr=args.lr, max_iter=10,history_size=20)
        elif args.optimizer == "adam":
            O = torch.optim.Adam(A.parameters(),lr=args.lr)
        elif args.optimizer == "sgd":
            O = torch.optim.SGD(A.parameters(),lr=args.lr, momentum=0.9)
        else:
            print("invalid optimizer name")
            return

        # target 

        
        # scheduler
        if args.scheduler=="exponential-lr":
            S = torch.optim.lr_scheduler.ExponentialLR(O,gamma=args.gamma)
        elif args.scheduler=="OneCycleLR":
            S = torch.optim.lr_scheduler.OneCycleLR(O,max_lr=args.lr,total_steps=args.n,pct_start=args.gamma)
        else:
            print("invalid scheduler")
            return
     
        if   args.loss == "acs":
            L = acs
        elif args.loss == "acs_target" or args.loss == "acs_none":
            L = acs_target
        elif args.loss == "acs_lgs":
            L = acs_lgs
        elif args.loss == "acs_ilp":
            L = acs_ilp
        elif args.loss == "aee_lgs":
            L = aee_lgs
        elif args.loss == "aee_ilp":
            L = aee_ilp
        elif args.loss == "aee_target":
            L = aee_target  
        elif args.loss == "mse":
            L = mse_target
        elif args.loss == "mse_lgs":
            L = mse_lgs
        elif args.loss == "mse_ilp":
            L = mse_ilp
        else:
            print("invalid loss function name")
            return
        
        args.output_folder = folder_path
        seed = args.seed if args.seed != -1 else np.random.randint(1000)
        print("Using seed", seed)
        log_param("seed", seed)
        A = train(A,
                    model,
                    D,
                    L,
                    data_loader,
                    O,
                    S,
                    args,
                    device,
                    seed=seed)

        if args.eval_after:
            print("Evaluating after training")
            from evaluate_patch_withDefense import evaluate_patch,change_arguments_from_runid
            vars(args)["run_id"] = run.info.run_id
            vars(args)["save_frequency"] = 1
            vars(args)["n"] = -1
            vars(args)["dataset_stage"] = 'evaluation'
            change_arguments_from_runid(args)
            mod_args = args
            eval_metrics = []
            evaluation = {}
            for patch_name in args.patch_name: # is made into a list with all patch_names to evaluate
                mod_args.patch_name = patch_name
                mod_args.nested = True
                evaluation = evaluate_patch(mod_args)
                eval_metrics.append(list(evaluation.values()))
                print("Evaluation metrics:", eval_metrics)
            eval_metrics = np.mean(eval_metrics,axis=0)
            names=list(evaluation.keys())
            print(names,eval_metrics)
            assert len(names) == len(eval_metrics), "Did you chaevaluationnge the number of metrics? Then you need to change the names here as well."+str(names)+str(eval_metrics)
            
            # log all values starting with aee_avg
            filtered_names = [name for name in names if name.startswith("aee_avg")]
            filtered_metrics = [eval_metrics[names.index(name)] for name in filtered_names]
            logging.log_metrics(args.n, *list(zip(filtered_names, filtered_metrics)))
            
            # log all values
            # logging.log_metrics(args.n, *list(zip(names, eval_metrics)))


if __name__ == "__main__":
    parser = parsing_file.create_parser(stage='training', attack_type='patch_attack_withDefense')
    args = parser.parse_args()

    # experiments = mlflow.list_experiments()
    # exp_id_name_pairs = [(exp.experiment_id, exp.name) for exp in experiments]
    # # create a dictionary with all experiment ids for each network and defense
    # net = args.net
    # dataset = args.dataset
    # # get all runs
    # runs = mlflow.search_runs(experiment_ids=[exp_id for exp_id, exp_name in exp_id_name_pairs if f"{net}_PatchAttack-with-defense_cd_u_{dataset[:6]}_" in exp_name and 'eval' not in exp_name])
    # # filter for finished runs with the same parameters
    # runs = runs[runs['status'] == 'FINISHED']
    # runs = runs[runs['params.dataset_name'] == args.dataset]
    # runs = runs[runs['params.model'] == args.net]
    # runs = runs[runs['params.defense'] == args.defense]
    # runs = runs[runs['params.seed'] == str(args.seed)]

    # if len(runs) > 0:
    #     print(f'Found {len(runs)} runs with parameters:')
    #     print(f'\tModel: {args.net}, \n\tDataset: {args.dataset}, \n\tDefense: {args.defense}, \n\tSeed: {args.seed}')
    #     exit()

    train_patch(args)
