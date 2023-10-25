from collections import OrderedDict
import pandas as pd
import sys
import os.path as op
import os
from tqdm import tqdm
import torch
import mlflow
from mlflow import log_param, log_metric, log_artifact
import numpy as np

from helper_functions.config_specs import Conf
from helper_functions import ownutilities, parsing_file, logging, targets
from helper_functions.defenses import LGS, ILP
from helper_functions.patch_adversary import PatchAdversary
from helper_functions.losses import aee_masked, acs_masked, mse_masked

class CustomLogger:
    """
    Custom logger for the evaluation of the patch attack with defense. Calculates the metrics and saves passes them to mlflow.
    """
    def __init__(self,use_target, n, no_save, output_folder, unregistered_artifacts) -> None:
        self.use_target = use_target
        self.n = n
        self.no_save = no_save
        self.output_folder = output_folder
        self.unregistered_artifacts = unregistered_artifacts
        self.i = 0

        # metrics:
        self.sum_aee_def_advdef = 0
        self.sum_aee_gt_advdef = 0
        self.sum_aee_gt_def = 0
        self.sum_aee_gt_undef = 0
        self.sum_aee_gt_adv = 0
        self.sum_aee_adv_advdef = 0
        self.sum_aee_adv_def = 0
        self.sum_aee_adv_undef = 0
        self.sum_aee_undef_advdef = 0
        self.sum_aee_undef_def = 0

        self.sum_mse_def_advdef = 0
        self.sum_mse_gt_advdef = 0
        self.sum_mse_gt_def = 0
        self.sum_mse_gt_undef = 0
        self.sum_mse_gt_adv = 0
        self.sum_mse_adv_advdef = 0
        self.sum_mse_adv_def = 0
        self.sum_mse_adv_undef = 0
        self.sum_mse_undef_advdef = 0
        self.sum_mse_undef_def = 0

        self.sum_acs_def_advdef = 0
        self.sum_acs_gt_advdef = 0
        self.sum_acs_gt_def = 0
        self.sum_acs_gt_undef = 0
        self.sum_acs_gt_adv = 0
        self.sum_acs_adv_advdef = 0
        self.sum_acs_adv_def = 0
        self.sum_acs_adv_undef = 0
        self.sum_acs_undef_advdef = 0
        self.sum_acs_undef_def = 0

        self.sum_aee_tgt_advdef = 0
        self.sum_aee_tgt_adv = 0
        self.sum_aee_tgt_def = 0
        self.sum_aee_tgt_undef = 0

        self.sum_mse_tgt_advdef = 0
        self.sum_mse_tgt_adv = 0
        self.sum_mse_tgt_def = 0
        self.sum_mse_tgt_undef = 0

        self.sum_acs_tgt_advdef = 0
        self.sum_acs_tgt_adv = 0
        self.sum_acs_tgt_def = 0
        self.sum_acs_tgt_undef = 0
    
    def update(self, I1, I2, I1_p, I2_p, I1_attacked_def, I2_attacked_def, I1_unattacked_def, I2_unattacked_def,
                F_unattacked, F_attacked, F_attacked_def, F_unattacked_def, F_gt, M, M_gt, target=None):
        aee_def_advdef = aee_masked(F_unattacked_def, F_attacked_def, (1-M)).cpu().item() # defended vs attack on defended
        aee_gt_advdef = aee_masked(F_gt, F_attacked_def, (1-M)*M_gt).cpu().item() if M_gt.any() else 0 # GroundTruth vs attack on defended
        aee_gt_def = aee_masked(F_gt, F_unattacked_def, (1-M)*M_gt).cpu().item() if M_gt.any() else 0 # GroundTruth vs defended
        aee_gt_undef = aee_masked(F_gt, F_unattacked, (1-M)*M_gt).cpu().item() if M_gt.any() else 0 # GroundTruth vs undefended
        aee_gt_adv = aee_masked(F_gt, F_attacked, (1-M)*M_gt).cpu().item() if M_gt.any() else 0 # GroundTruth vs attack on undefended
        aee_adv_advdef = aee_masked(F_attacked, F_attacked_def, (1-M)).cpu().item() # attack on undefended vs attack on defended
        aee_adv_def = aee_masked(F_attacked, F_unattacked_def, (1-M)).cpu().item() # attack on undefended vs defended
        aee_adv_undef = aee_masked(F_attacked, F_unattacked, (1-M)).cpu().item() # attack on undefended vs undefended
        aee_undef_advdef = aee_masked(F_unattacked, F_attacked_def, (1-M)).cpu().item() # undefended vs attack on defended
        aee_undef_def = aee_masked(F_unattacked, F_unattacked_def, (1-M)).cpu().item() # undefended vs defended

        mse_def_advdef = mse_masked(F_unattacked_def, F_attacked_def, (1-M)).cpu().item() # defended vs attack on defended
        mse_gt_advdef = mse_masked(F_gt, F_attacked_def, (1-M)*M_gt).cpu().item() if M_gt.any() else 0 # GroundTruth vs attack on defended
        mse_gt_def = mse_masked(F_gt, F_unattacked_def, (1-M)*M_gt).cpu().item() if M_gt.any() else 0 # GroundTruth vs defended
        mse_gt_undef = mse_masked(F_gt, F_unattacked, (1-M)*M_gt).cpu().item() if M_gt.any() else 0 # GroundTruth vs undefended
        mse_gt_adv = mse_masked(F_gt, F_attacked, (1-M)*M_gt).cpu().item() if M_gt.any() else 0 # GroundTruth vs attack on undefended
        mse_adv_advdef = mse_masked(F_attacked, F_attacked_def, (1-M)).cpu().item() # attack on undefended vs attack on defended
        mse_adv_def = mse_masked(F_attacked, F_unattacked_def, (1-M)).cpu().item() # attack on undefended vs defended
        mse_adv_undef = mse_masked(F_attacked, F_unattacked, (1-M)).cpu().item() # attack on undefended vs undefended
        mse_undef_advdef = mse_masked(F_unattacked, F_attacked_def, (1-M)).cpu().item() # undefended vs attack on defended
        mse_undef_def = mse_masked(F_unattacked, F_unattacked_def, (1-M)).cpu().item() # undefended vs defended

        acs_def_advdef = acs_masked(F_unattacked_def, F_attacked_def, (1-M)).cpu().item() # defended vs attack on defended
        acs_gt_advdef = acs_masked(F_gt, F_attacked_def, (1-M)*M_gt).cpu().item() if M_gt.any() else 0 # GroundTruth vs attack on defended
        acs_gt_def = acs_masked(F_gt, F_unattacked_def, (1-M)*M_gt).cpu().item() if M_gt.any() else 0 # GroundTruth vs defended
        acs_gt_undef = acs_masked(F_gt, F_unattacked, (1-M)*M_gt).cpu().item() if M_gt.any() else 0 # GroundTruth vs undefended
        acs_gt_adv = acs_masked(F_gt, F_attacked, (1-M)*M_gt).cpu().item() if M_gt.any() else 0 # GroundTruth vs attack on undefended
        acs_adv_advdef = acs_masked(F_attacked, F_attacked_def, (1-M)).cpu().item() # attack on undefended vs attack on defended
        acs_adv_def = acs_masked(F_attacked, F_unattacked_def, (1-M)).cpu().item() # attack on undefended vs defended
        acs_adv_undef = acs_masked(F_attacked, F_unattacked, (1-M)).cpu().item() # attack on undefended vs undefended
        acs_undef_advdef = acs_masked(F_unattacked, F_attacked_def, (1-M)).cpu().item() # undefended vs attack on defended
        acs_undef_def = acs_masked(F_unattacked, F_unattacked_def, (1-M)).cpu().item() # undefended vs defended

        if self.use_target:
            aee_tgt_advdef = aee_masked(target, F_attacked_def, (1-M)).cpu().item()
            aee_tgt_adv = aee_masked(target, F_attacked, (1-M)).cpu().item()
            aee_tgt_def = aee_masked(target, F_unattacked_def, (1-M)).cpu().item()
            aee_tgt_undef = aee_masked(target, F_unattacked, (1-M)).cpu().item()

            mse_tgt_advdef = mse_masked(target, F_attacked_def, (1-M)).cpu().item()
            mse_tgt_adv = mse_masked(target, F_attacked, (1-M)).cpu().item()
            mse_tgt_def = mse_masked(target, F_unattacked_def, (1-M)).cpu().item()
            mse_tgt_undef = mse_masked(target, F_unattacked, (1-M)).cpu().item()

            acs_tgt_advdef = acs_masked(target, F_attacked_def, (1-M)).cpu().item()
            acs_tgt_adv = acs_masked(target, F_attacked, (1-M)).cpu().item()
            acs_tgt_def = acs_masked(target, F_unattacked_def, (1-M)).cpu().item()
            acs_tgt_undef = acs_masked(target, F_unattacked, (1-M)).cpu().item()

        self.sum_aee_def_advdef += aee_def_advdef
        self.sum_aee_gt_advdef += aee_gt_advdef
        self.sum_aee_gt_def += aee_gt_def
        self.sum_aee_gt_undef += aee_gt_undef
        self.sum_aee_gt_adv += aee_gt_adv
        self.sum_aee_adv_advdef += aee_adv_advdef
        self.sum_aee_adv_def += aee_adv_def
        self.sum_aee_adv_undef += aee_adv_undef
        self.sum_aee_undef_advdef += aee_undef_advdef
        self.sum_aee_undef_def += aee_undef_def

        self.sum_mse_def_advdef += mse_def_advdef
        self.sum_mse_gt_advdef += mse_gt_advdef
        self.sum_mse_gt_def += mse_gt_def
        self.sum_mse_gt_undef += mse_gt_undef
        self.sum_mse_gt_adv += mse_gt_adv
        self.sum_mse_adv_advdef += mse_adv_advdef
        self.sum_mse_adv_def += mse_adv_def
        self.sum_mse_adv_undef += mse_adv_undef
        self.sum_mse_undef_advdef += mse_undef_advdef
        self.sum_mse_undef_def += mse_undef_def

        self.sum_acs_def_advdef += acs_def_advdef
        self.sum_acs_gt_advdef += acs_gt_advdef
        self.sum_acs_gt_def += acs_gt_def
        self.sum_acs_gt_undef += acs_gt_undef
        self.sum_acs_gt_adv += acs_gt_adv
        self.sum_acs_adv_advdef += acs_adv_advdef
        self.sum_acs_adv_def += acs_adv_def
        self.sum_acs_adv_undef += acs_adv_undef
        self.sum_acs_undef_advdef += acs_undef_advdef
        self.sum_acs_undef_def += acs_undef_def

        if self.use_target:
            self.sum_aee_tgt_advdef += aee_tgt_advdef
            self.sum_aee_tgt_adv += aee_tgt_adv
            self.sum_aee_tgt_def += aee_tgt_def
            self.sum_aee_tgt_undef += aee_tgt_undef

            self.sum_mse_tgt_advdef += mse_tgt_advdef
            self.sum_mse_tgt_adv += mse_tgt_adv
            self.sum_mse_tgt_def += mse_tgt_def
            self.sum_mse_tgt_undef += mse_tgt_undef

            self.sum_acs_tgt_advdef += acs_tgt_advdef
            self.sum_acs_tgt_adv += acs_tgt_adv
            self.sum_acs_tgt_def += acs_tgt_def
            self.sum_acs_tgt_undef += acs_tgt_undef

        logging.log_metrics(self.i,  ("aee_def-advdef", aee_def_advdef),
                                ("aee_gt-advdef", aee_gt_advdef),
                                ("aee_gt-def", aee_gt_def),
                                ("aee_gt-undef", aee_gt_undef),
                                ("aee_gt-adv", aee_gt_adv),
                                ("aee_adv-advdef", aee_adv_advdef),
                                ("aee_adv-def", aee_adv_def),
                                ("aee_adv-undef", aee_adv_undef),
                                ("aee_undef-advdef", aee_undef_advdef),
                                ("aee_undef-def", aee_undef_def),
                                ("mse_def-advdef", mse_def_advdef),
                                ("mse_gt-advdef", mse_gt_advdef),
                                ("mse_gt-def", mse_gt_def),
                                ("mse_gt-undef", mse_gt_undef),
                                ("mse_gt-adv", mse_gt_adv),
                                ("mse_adv-advdef", mse_adv_advdef),
                                ("mse_adv-def", mse_adv_def),
                                ("mse_adv-undef", mse_adv_undef),
                                ("mse_undef-advdef", mse_undef_advdef),
                                ("mse_undef-def", mse_undef_def),
                                ("acs_def-advdef", acs_def_advdef),
                                ("acs_gt-advdef", acs_gt_advdef),
                                ("acs_adv-advdef", acs_adv_advdef),
                                ("acs_adv-def", acs_adv_def),
                                ("acs_adv-undef", acs_adv_undef),
                                ("acs_undef-advdef", acs_undef_advdef),
                                ("acs_undef-def", acs_undef_def))
        if self.use_target:
            logging.log_metrics(self.i,  ("aee_tgt-advdef", aee_tgt_advdef),
                                    ("aee_tgt-adv", aee_tgt_adv),
                                    ("aee_tgt-def", aee_tgt_def),
                                    ("aee_tgt-undef", aee_tgt_undef),
                                    ("mse_tgt-advdef", mse_tgt_advdef),
                                    ("mse_tgt-adv", mse_tgt_adv),
                                    ("mse_tgt-def", mse_tgt_def),
                                    ("mse_tgt-undef", mse_tgt_undef),
                                    ("acs_tgt-advdef", acs_tgt_advdef),
                                    ("acs_tgt-adv", acs_tgt_adv),
                                    ("acs_tgt-def", acs_tgt_def),
                                    ("acs_tgt-undef", acs_tgt_undef))
        mlflow.log_metric("current_iteration", self.i)

        # only if in the last step or in line with the save frequency
        if self.i==self.n-1 and not self.no_save:
            logging.save_image(I1_attacked_def, self.i, self.output_folder, image_name="I1_attacked_def", unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_image(I2_attacked_def, self.i, self.output_folder, image_name="I2_attacked_def", unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_image(I1_unattacked_def, self.i, self.output_folder, image_name="I1_unattacked_def", unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_image(I2_unattacked_def, self.i, self.output_folder, image_name="I2_unattacked_def", unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_image(I1_p, self.i, self.output_folder, image_name="I1_attacked", unit_input=True, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_image(I2_p, self.i, self.output_folder, image_name="I2_attacked", unit_input=True, unregistered_artifacts=self.unregistered_artifacts)

            max_flow_gt = 0
            if M_gt.any(): # aka if ground truth is available
                max_flow_gt = ownutilities.maximum_flow(F_gt)
            max_flow = np.max([max_flow_gt, 
                            ownutilities.maximum_flow(F_unattacked_def), 
                            ownutilities.maximum_flow(F_attacked_def),
                            ownutilities.maximum_flow(F_attacked),
                            ownutilities.maximum_flow(F_unattacked)])

            logging.save_flow(F_attacked_def, self.i, self.output_folder, flow_name='flow_advdef', auto_scale=False, max_scale=max_flow, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_tensor(F_attacked_def, "flow_advdef", self.i, self.output_folder, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_flow(F_unattacked_def, self.i, self.output_folder, flow_name='flow_def', auto_scale=False, max_scale=max_flow, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_tensor(F_unattacked_def, "flow_def", self.i, self.output_folder, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_flow(F_attacked, self.i, self.output_folder, flow_name='flow_adv', auto_scale=False, max_scale=max_flow, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_tensor(F_attacked, "flow_adv", self.i, self.output_folder, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_flow(F_unattacked, self.i, self.output_folder, flow_name='flow_unattacked', auto_scale=False, max_scale=max_flow, unregistered_artifacts=self.unregistered_artifacts)
            logging.save_tensor(F_unattacked, "flow_unattacked", self.i, self.output_folder, unregistered_artifacts=self.unregistered_artifacts)
            if M_gt.any():
                logging.save_flow(F_gt, self.i, self.output_folder, flow_name='flow_gt', auto_scale=False, max_scale=max_flow, unregistered_artifacts=self.unregistered_artifacts)
                logging.save_tensor(F_gt, "flow_gt", self.i, self.output_folder, unregistered_artifacts=self.unregistered_artifacts)
        
        self.i += 1 

    def close(self):
        avgs = logging.calc_log_averages(self.i,
                                ("aee_avg_def-advdef", self.sum_aee_def_advdef),
                                ("aee_avg_gt-advdef", self.sum_aee_gt_advdef),
                                ("aee_avg_gt-def", self.sum_aee_gt_def),
                                ("aee_avg_gt-undef", self.sum_aee_gt_undef),
                                ("aee_avg_gt-adv", self.sum_aee_gt_adv),
                                ("aee_avg_adv-advdef", self.sum_aee_adv_advdef),
                                ("aee_avg_adv-def", self.sum_aee_adv_def),
                                ("aee_avg_adv-undef", self.sum_aee_adv_undef),
                                ("aee_avg_undef-advdef", self.sum_aee_undef_advdef),
                                ("aee_avg_undef-def", self.sum_aee_undef_def),
                                ("mse_avg_def-advdef", self.sum_mse_def_advdef),
                                ("mse_avg_gt-advdef", self.sum_mse_gt_advdef),
                                ("mse_avg_gt-def", self.sum_mse_gt_def),
                                ("mse_avg_gt-undef", self.sum_mse_gt_undef),
                                ("mse_avg_gt-adv", self.sum_mse_gt_adv),
                                ("mse_avg_adv-advdef", self.sum_mse_adv_advdef),
                                ("mse_avg_adv-def", self.sum_mse_adv_def),
                                ("mse_avg_adv-undef", self.sum_mse_adv_undef),
                                ("mse_avg_undef-advdef", self.sum_mse_undef_advdef),
                                ("mse_avg_undef-def", self.sum_mse_undef_def),
                                ("acs_avg_def-advdef", self.sum_acs_def_advdef),
                                ("acs_avg_gt-advdef", self.sum_acs_gt_advdef),
                                ("acs_avg_gt-def", self.sum_acs_gt_def),
                                ("acs_avg_gt-undef", self.sum_acs_gt_undef),
                                ("acs_avg_gt-adv", self.sum_acs_gt_adv),
                                ("acs_avg_adv-advdef", self.sum_acs_adv_advdef),
                                ("acs_avg_adv-def", self.sum_acs_adv_def),
                                ("acs_avg_adv-undef", self.sum_acs_adv_undef),
                                ("acs_avg_undef-advdef", self.sum_acs_undef_advdef),
                                ("acs_avg_undef-def", self.sum_acs_undef_def),
                                ("aee_avg_tgt-advdef", self.sum_aee_tgt_advdef),
                                ("aee_avg_tgt-adv", self.sum_aee_tgt_adv),
                                ("aee_avg_tgt-def", self.sum_aee_tgt_def),
                                ("aee_avg_tgt-undef", self.sum_aee_tgt_undef),
                                ("mse_avg_tgt-advdef", self.sum_mse_tgt_advdef),
                                ("mse_avg_tgt-adv", self.sum_mse_tgt_adv),
                                ("mse_avg_tgt-def", self.sum_mse_tgt_def),
                                ("mse_avg_tgt-undef", self.sum_mse_tgt_undef),
                                ("acs_avg_tgt-advdef", self.sum_acs_tgt_advdef),
                                ("acs_avg_tgt-adv", self.sum_acs_tgt_adv),
                                ("acs_avg_tgt-def", self.sum_acs_tgt_def),
                                ("acs_avg_tgt-undef", self.sum_acs_tgt_undef))
        return OrderedDict({k: v for k, v in zip(["aee_avg_def-advdef", "aee_avg_gt-advdef", "aee_avg_gt-def", "aee_avg_gt-undef", "aee_avg_gt-adv", "aee_avg_adv-advdef", "aee_avg_adv-def", "aee_avg_adv-undef", "aee_avg_undef-advdef", "aee_avg_undef-def", "mse_avg_def-advdef", "mse_avg_gt-advdef", "mse_avg_gt-def", "mse_avg_gt-undef", "mse_avg_gt-adv", "mse_avg_adv-advdef", "mse_avg_adv-def", "mse_avg_adv-undef", "mse_avg_undef-advdef", "mse_avg_undef-def", "acs_avg_def-advdef", "acs_avg_gt-advdef", "acs_avg_gt-def", "acs_avg_gt-undef", "acs_avg_gt-adv", "acs_avg_adv-advdef", "acs_avg_adv-def", "acs_avg_adv-undef", "acs_avg_undef-advdef", "acs_avg_undef-def", "aee_avg_tgt-advdef", "aee_avg_tgt-adv", "aee_avg_tgt-def", "aee_avg_tgt-undef", "mse_avg_tgt-advdef", "mse_avg_tgt-adv", "mse_avg_tgt-def", "mse_avg_tgt-undef", "acs_avg_tgt-advdef", "acs_avg_tgt-adv", "acs_avg_tgt-def", "acs_avg_tgt-undef"], avgs)})

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

@torch.no_grad()
def evaluate(A, N, D, dl, n, args, device='cuda'):
    seed = max(args.seed,0)
    print("Seed:", seed)
    if args.seed is not None:
        import random
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False # may reduce perfomance (https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking)
        torch.backends.cudnn.deterministic=True


    use_target = (args.target is not None) or (args.target != "None")
    # for logging
    unit_images = ownutilities.model_takes_unit_input(args.net)
    metrics = CustomLogger(use_target=use_target, n=args.n, no_save=args.no_save, output_folder=args.output_folder, unregistered_artifacts=args.unregistered_artifacts)

    try:
        n = min(n, len(dl))
    except:
        pass
    for i, data in enumerate(tqdm(dl, total=n)):
        if i == n:
            break
        I1, I2, F_gt, M_gt = [x.to(device) for x in data]  # M_gt is the mask and False if no ground truth is available
        if not unit_images:
            # If the model takes unit input, ownutilities.preprocess_img will transform images into [0,1].
            # Otherwise, do transformation here
            I1 = I1/255.
            I2 = I2/255.
        padder, [I1, I2] = ownutilities.preprocess_img(args.net, I1, I2)

        # 1.) Attack
        I1_p, I2_p, M, y, x = A(I1, I2)
        [M] = padder.unpad(M).to(device)

        # 2.) Defense
        if D is None:
            (I1_unattacked_def, I2_unattacked_def) = (I1, I2)  # if D is None else
            (I1_attacked_def, I2_attacked_def) = (I1_p, I2_p)  # if D is None else
        else:
            (I1_unattacked_def, I2_unattacked_def) = D(I1, I2)
            (I1_attacked_def, I2_attacked_def) = D(I1_p, I2_p)

        # 3.) Prediction
        F_attacked_def = ownutilities.compute_flow(N, "scaled_input_model", I1_attacked_def, I2_attacked_def)
        [F_attacked_def] = ownutilities.postprocess_flow(args.net, padder, F_attacked_def)
        F_unattacked_def = ownutilities.compute_flow(N, "scaled_input_model", I1_unattacked_def, I2_unattacked_def)
        [F_unattacked_def] = ownutilities.postprocess_flow(args.net, padder, F_unattacked_def)
        # only for evaluation
        F_orig = ownutilities.compute_flow(N, "scaled_input_model", I1, I2)
        [F_orig] = ownutilities.postprocess_flow(args.net, padder, F_orig)
        F_attacked_undef = ownutilities.compute_flow(N, "scaled_input_model", I1_p, I2_p)
        [F_attacked_undef] = ownutilities.postprocess_flow(args.net, padder, F_attacked_undef)

        if use_target:
            target = targets.get_target(args.target, F_unattacked_def, flow_target_scale=args.flow_target_scale, custom_target_path=args.custom_target_path, device=device)
        else: 
            target = None
        # 4.) Evaluation and logging
        metrics.update(I1, I2, I1_p, I2_p, I1_attacked_def, I2_attacked_def, I1_unattacked_def, I2_unattacked_def,
                    F_orig, F_attacked_undef, F_attacked_def, F_unattacked_def, F_gt, M, M_gt, target)
    return metrics.close()


def change_arguments_from_runid(args):
    run = mlflow.get_run(args.run_id)
    print("Loading arguments from MLFlow:",f"127.0.0.1:5000/#/experiments/{run.info._experiment_id}/runs/{args.run_id}")

    params = run.data.params
    args_dict = vars(args)
    # The following loads parameters where the names match. The parameters patch_name, custom_weight_path are loaded if not specified
    for key in ["k", "o", "t", "s", "r", "alpha", "defense","patch_size","optimizer","change_of_variables", "crop_shape","scheduler","gamma", "lr", "n_patches_eval", "no_save"]:
        assert key in params, f"Parameter {key} not found in run {args.run_id}"
        # load with correct type:
        if str.isnumeric(params[key].replace('.','')):
            if float(params[key]).is_integer():
                args_dict[key] = int(args_dict[key])
            else:
                args_dict[key] = float(params[key])
        else:
            args_dict[key] = params[key]
    
    args_dict["loss"]=params["loss"]
    if 'acs' not in params['loss']:
        args_dict["custom_target_path"] = params["custom_target_path"]
        args_dict["flow_target_scale"] = float(params["flow_target_scale"])
        args_dict["target"] = params["attack_target"]
    else:
        args_dict["target"] = 'neg_flow'

    # last_iter = int(run.data.metrics['last_saved'])
    if args.patch_name=='':
        args_dict['patch_name'] = []
        for i in range(int(args.n_patches_eval)):
            # load the last saved artifact from the run
            artifact_folder = mlflow.artifacts.download_artifacts(run.info.artifact_uri) # load artifact folder
            args_dict['patch_name'].append(os.path.join(artifact_folder,f'{int(params["n"])+i*15-1:05d}_Patch.npy')) # add the last patch to the path
    if args.custom_weight_path=='':
        args_dict['custom_weight_path'] = params['model_path_weights']
    
    return args


def evaluate_patch(args):

    if Conf.config('useCPU') or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Setting Device to {device}")

    try:
        args.nested
        nested = True
    except AttributeError:
        nested = False
    experiment_id, folder_path, folder_name = logging.mlflow_experimental_setup(args.output_folder, args.net, "PatchAttack-with-defense", True, True, args.custom_experiment_name, stage="eval")
    with mlflow.start_run(experiment_id=experiment_id, run_name=folder_name, nested = nested) as run:

        print("\nStarting Defended Patch Evaluation:")
        print()
        print("\tModel:                   %s" % (args.net))
        print("\tDataset:                 %s" % (args.dataset))
        print("\tDefense:                 %s" % (args.defense))
        print("\tPatch:                   %s" % (args.patch_name))
        
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


        ## logging 
        model_takes_unit_input = ownutilities.model_takes_unit_input(args.net)
        logging.log_model_params(args.net,model_takes_unit_input)
        logging.log_dataset_params(args.dataset, 1, 1, args.dstype, args.dataset_stage)
        logging.log_attack_params("PatchAttack-with-defense", args.loss, args.target, True, True)
        log_param("patch_name", args.patch_name)
        log_param("patch_size", args.patch_size)
        log_param("defense", args.defense)
        log_param("optimizer", args.optimizer)
        log_param("loss",args.loss)
        log_param("lr", args.lr)
        log_param("scheduler", args.scheduler)
        log_param("gamma", args.gamma)
        log_param("crop_shape", args.crop_shape)
        log_param("k", args.k)
        log_param("o", args.o)
        log_param("t", args.t)
        log_param("s", args.s)
        log_param("r", args.r)
        log_param("n", args.n)
        log_param("alpha", args.alpha)
        log_param("max_delta", args.max_delta)
        log_param("run_id", args.run_id)

        # dataset
        data_loader, has_gt = ownutilities.prepare_dataloader(args.dataset_stage,
                                                              dataset=args.dataset,
                                                              shuffle=False,
                                                              # sintel specific:
                                                              dstype=args.dstype)

        # Flownet
        model_takes_unit_input = ownutilities.model_takes_unit_input(args.net)
        model, path_weights = ownutilities.import_and_load(args.net, custom_weight_path=args.custom_weight_path,
                                                           make_unit_input=not model_takes_unit_input, variable_change=False, make_scaled_input_model=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()

        # defense
        if args.defense == "lgs":
            D = LGS(args.k, args.o, args.t, args.s, "forward")
        elif args.defense == "ilp":
            D = ILP(args.k, args.o, args.t, args.s, args.r, "forward")
        elif args.defense == "none":
            D = None
        else:
            print("invalid defense name")
            return

        print("Loading patch from", args.patch_name)
        A = PatchAdversary(args.patch_name,size=args.patch_size,angle=[-10,10],scale=[0.95,1.05], change_of_variable=False).to(device)
        logging.save_tensor(A.get_P(), f"Patch",-1,args.output_folder,args.unregistered_artifacts)
        logging.save_image(A.get_P(Mask=True),-1,args.output_folder,image_name="Patch",unit_input=True, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_image(A.get_P(),-1,args.output_folder,image_name="Patch_no_mask",unit_input=True, unregistered_artifacts=args.unregistered_artifacts)

        return evaluate(A, model, D, data_loader, args.n, args, device)



if __name__ == "__main__":
    parser = parsing_file.create_parser(stage='evaluation', attack_type='patch_attack_withDefense')
    args = parser.parse_args()

    if args.run_id:
        args = change_arguments_from_runid(args)

    print(evaluate_patch(args))
