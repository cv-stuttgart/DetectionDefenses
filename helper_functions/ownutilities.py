import json
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
import os
import sys
#required to prevent ModuleNotFoundError for 'flow_plot'. The flow_library is a submodule, which imports its own functions and can therefore not be imported with flow_library.flow_plot
sys.path.append("flow_library")


from PIL import Image

from torch.utils.data import DataLoader, Subset
from helper_functions import datasets
from helper_functions.config_specs import Paths, Conf


class InputPadder:
    """Pads images such that dimensions are divisible by divisor

    This method is taken from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    """
    def __init__(self, dims, divisor=8, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        """Pad a batch of input images such that the image size is divisible by the factor specified as divisor

        Returns:
            list: padded input images
        """
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def get_dimensions(self):
        """get the original spatial dimension of the image

        Returns:
            int: original image height and width
        """
        return self.ht, self.wd

    def unpad(self,x):
        """undo the padding and restore original spatial dimension

        Args:
            x (tensor): a tensor with padded dimensions

        Returns:
            tesnor: tensor with removed padding (i.e. original spatial dimension)
        """
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def import_and_load(net='RAFT', make_unit_input=False, variable_change=False, device=torch.device("cpu"), make_scaled_input_model=False, **kwargs):
    """import a model and load pretrained weights for it

    Args:
        net (str, optional):
            the desired network to load. Defaults to 'RAFT'.
        make_unit_input (bool, optional):
            model will assume input images in range [0,1] and transform to [0,255]. Defaults to False.
        variable_change (bool, optional):
            apply change of variables (COV). Defaults to False.
        device (torch.device, optional):
            changes the selected device. Defaults to torch.device("cpu").
        make_scaled_input_model (bool, optional):
            load a scaled input model which uses make_unit_input and variable_change as specified. Defaults to False.

    Raises:
        RuntimeWarning: Unknown model type

    Returns:
        torch.nn.Module: PyTorch optical flow model with loaded weights
    """

    if make_unit_input==True or variable_change==True or make_scaled_input_model:
        from helper_functions.own_models import ScaledInputModel
        model = ScaledInputModel(net, make_unit_input=make_unit_input, variable_change=variable_change, device=device, **kwargs)
        print("--> transforming model to 'make_unit_input'=%s, 'variable_change'=%s\n" % (str(make_unit_input), str(variable_change)))
        path_weights = model.return_path_weights()

    else:
        model = None
        path_weights = ""
        custom_weight_path = kwargs["custom_weight_path"] if "custom_weight_path" in kwargs else ""
        try:
            if net == 'RAFT':
                from models.raft.raft import RAFT

                # set the path to the corresponding weights for initializing the model
                path_weights = custom_weight_path or 'models/_pretrained_weights/raft-sintel.pth'

                # possible adjustements to the config can be made in the file
                # found in the same directory as _pretrained_weights
                path_config = os.path.join(*path_weights.split(os.path.sep)[:-2], "_config", "raft_config.json")
                with open(path_config) as file:
                    config = json.load(file)

                model = torch.nn.DataParallel(RAFT(config))
                # load pretrained weights
                model.load_state_dict(torch.load(path_weights, map_location=device))

            elif net == 'GMA':
                from models.gma.network import RAFTGMA

                # set the path to the corresponding weights for initializing the model
                path_weights = custom_weight_path or 'models/_pretrained_weights/gma-sintel.pth'

                # possible adjustements to the config file can be made
                # under models/_config/gma_config.json
                with open("models/_config/gma_config.json") as file:
                    config = json.load(file)
                    # GMA accepts only a Namespace object when initializing
                    config = Namespace(**config)

                model = torch.nn.DataParallel(RAFTGMA(config))

                model.load_state_dict(torch.load(path_weights, map_location=device))

            elif net == "FlowFormer":
                from models.FlowFormer.core.FlowFormer import build_flowformer
                from models.FlowFormer.configs.things_eval import get_cfg as get_things_cfg

                path_weights = custom_weight_path or 'models/_pretrained_weights/flowformer_weights/sintel.pth'
                cfg = get_things_cfg()
                model_args = Namespace(model=path_weights, mixed_precision=False, alternate_corr=False)
                cfg.update(vars(model_args))

                model = torch.nn.DataParallel(build_flowformer(cfg))
                model.load_state_dict(torch.load(cfg.model, map_location=torch.device('cpu')))


            elif net =='PWCNet':
                from models.PWCNet.PWCNet import PWCDCNet

                # set path to pretrained weights:
                path_weights = custom_weight_path or 'models/_pretrained_weights/pwc_net_chairs.pth.tar'
                with warnings.catch_warnings():
                    # this will catch the deprecated warning for spynet and pwcnet to avoid messy console
                    warnings.simplefilter("ignore", UserWarning)
                    model = PWCDCNet()

                weights = torch.load(path_weights, map_location=device)
                if 'state_dict' in weights.keys():
                    model.load_state_dict(weights['state_dict'])
                else:
                    model.load_state_dict(weights)
                model.to(device)

            elif net =='SpyNet':
                from models.SpyNet.SpyNet import Network as SpyNet
                # weights for SpyNet are loaded during initialization
                model = SpyNet(nlevels=6, pretrained=True)
                model.to(device)

            elif net[:8] == "FlowNet2":
                # hard coding configuration for FlowNet2
                args_fn = Namespace(fp16=False, rgb_max=255.0)

                if net == "FlowNet2":
                    from models.FlowNet.FlowNet2 import FlowNet2
                    # set path to pretrained weights
                    path_weights = custom_weight_path or 'models/_pretrained_weights/FlowNet2_checkpoint.pth.tar'
                    model = FlowNet2(args_fn, div_flow=20, batchNorm=False)

                elif net == "FlowNet2S":
                    from models.FlowNet.FlowNet2S import FlowNet2S
                    # set path to pretrained weights
                    path_weights = custom_weight_path or 'models/_pretrained_weights/FlowNet2-S_checkpoint.pth.tar'
                    model = FlowNet2S(args_fn, div_flow=20, batchNorm=False)

                elif net == "FlowNet2C":
                    from models.FlowNet.FlowNet2C import FlowNet2C
                    # set path to pretrained weights
                    path_weights = custom_weight_path or 'models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar'
                    model = FlowNet2C(args_fn, div_flow=20, batchNorm=False)

                else:
                    raise ValueError("Unknown FlowNet2 type: %s" % (net))


                weights = torch.load(path_weights, map_location=device)
                model.load_state_dict(weights['state_dict'])

                model.to(device)

            elif net == "FlowNetCRobust":
                from models.FlowNetCRobust.FlowNetC_flexible_larger_field import FlowNetC_flexible_larger_field

                # initialize model and load pretrained weights
                path_weights = custom_weight_path or 'models/_pretrained_weights/RobustFlowNetC.pth'
                model = FlowNetC_flexible_larger_field(kernel_size=3, number_of_reps=3)
                
                weights = torch.load(path_weights, map_location=device)
                model.load_state_dict(weights)

                model.to(device)

            elif net[:7] == "FlowNet":
                # hard coding configuration for FlowNet
                args_fn = Namespace(fp16=False, rgb_max=255.0)

                if net == "FlowNetC":
                    # set path to pretrained weights
                    path_weights = custom_weight_path or 'models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar'
                    print("Loading FlowNetC weights from: %s" % (path_weights) )
                    from models.FlowNetC.FlowNetC import FlowNetC
                    model = FlowNetC(div_flow=20, batchNorm=False)
                    # from models.FlowNet.FlowNetC import FlowNetC
                    # model = FlowNetC(args_fn,div_flow=20, batchNorm=False)

                else:
                    raise ValueError("Unknown FlowNet type: %s" % (net))

                weights = torch.load(path_weights, map_location=device)
                model.load_state_dict(weights['state_dict'])

                model.to(device)

            if model is None:
                raise RuntimeWarning('The network %s is not a valid model option for import_and_load(network). No model was loaded. Use "RAFT", "GMA", "FlowNetC", "PWCNet" or "SpyNet" instead.' % (net))
        except FileNotFoundError as e:
            print("\nLoading the model failed, because the checkpoint path was invalid. Are the checkpoints placed in models/_pretrained_weights/? If this folder is empty, consider to execute the checkpoint loading script from scripts/load_all_weights.sh. The full error that caused the loading failure is below:\n\n%s" % e)
            exit()

        print("--> flow network is set to %s" % net)
    return model, path_weights

def prepare_dataloader(mode='training', dataset='Sintel', shuffle=False, batch_size=1, small_run=False, sintel_subsplit=False, dstype='clean', num_repeats=1):
    """Get a PyTorch dataloader for the specified dataset

    Args:
        mode (str, optional):
            Specify the split of the dataset [training | evaluation]. Defaults to 'training'.
        dataset (str, optional):
            Specify the dataset used [Sintel | Kitti15]. Defaults to 'Sintel'.
        shuffle (bool, optional):
            Use random sampling. Defaults to False.
        batch_size (int, optional):
            Defaults to 1.
        small_run (bool, optional):
            For debugging: Will load only 32 images. Defaults to False.
        dstype (str, optional):
            Specific for Sintel dataset. Dataset type [clean | final] . Defaults to 'clean'.

    Raises:
        ValueError: Unknown mode.
        ValueError: Unkown dataset.

    Returns:
        torch.utils.data.DataLoader: Dataloader which can be used for FGSM.
    """

    if dataset == 'Sintel':
        if not sintel_subsplit:
            if mode == 'training':
                dataset = datasets.MpiSintel(split=Paths.splits("sintel_train"),
                    root=Paths.config("sintel_mpi"), dstype=dstype, has_gt=True)
            elif mode == 'evaluation':
                dataset = datasets.MpiSintel(split=Paths.splits("sintel_eval"),
                    root=Paths.config("sintel_mpi"), dstype=dstype, has_gt=True) # gt from validation split
            else:
                raise ValueError(f'The specified mode: {mode} is unknown.')
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == 'SintelSplitZhao':
        # Sintel-train and validation split from S. Zhao et al. "MaskFlownet: Asymmetric feature matching with learnable occlusion mask" (CVPR 2020)
        if mode == 'training':
            dataset = datasets.MpiSintel(split=Paths.splits("sintel_train"), root=Paths.config("sintel_mpi"), dstype=dstype, has_gt=True,
            scenes=["alley_1","alley_2","ambush_4","ambush_5","ambush_7","bamboo_1","bandage_1","bandage_2","cave_2","market_2","market_5","mountain_1","shaman_2","shaman_3","sleeping_1","sleeping_2","temple_3"])
        elif mode == 'evaluation':
            dataset = datasets.MpiSintel(split=Paths.splits("sintel_train"), root=Paths.config("sintel_mpi"), dstype=dstype, has_gt=True,
            scenes=["ambush_2","ambush_6","bamboo_2","cave_4","market_6","temple_2"])
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == 'SintelSplitYang':
        # Sintel-train and validation split (in-distribution) from G. Yang et al. "High-resolution optical flow from 1D attention and correlation" (ICCV 2021)
        if mode == 'training':
            dataset = datasets.MpiSintel(split=Paths.splits("sintel_train"), root=Paths.config("sintel_mpi"), dstype=dstype, has_gt=True,
            scenes=["alley_1","alley_2","ambush_4","ambush_5","ambush_6","ambush_7","bamboo_1","bandage_1","bandage_2","cave_4","market_5","market_6","mountain_1","shaman_3","sleeping_1","sleeping_2","temple_3"])
        elif mode == 'evaluation':
            dataset = datasets.MpiSintel(split=Paths.splits("sintel_train"), root=Paths.config("sintel_mpi"), dstype=dstype, has_gt=True,
            scenes=["ambush_2", "bamboo_2", "cave_2", "market_2", "shaman_2", "temple_2"], every_nth_img=3)
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')


    elif dataset == 'Kitti15':
        # The KITTI15 dataset from M. Menze et al. "Object scene flow for autonomous vehicles" (CVPR 2015)
        if mode == 'training':
            dataset = datasets.KITTI(split=Paths.splits("kitti_train"), aug_params=None, root=Paths.config("kitti15"), has_gt=True)
        elif mode == 'evaluation':
            dataset = datasets.KITTI(split=Paths.splits("kitti_eval"), aug_params=None, root=Paths.config("kitti15"), has_gt=False)
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == "KittiRaw":
        if mode == 'training':
            dataset = datasets.KITTIRaw(split='training',root=Paths.config("kitti_raw"), has_gt=False)
        else:
            raise ValueError("'testing' not yet implementet for KittiRaw")

    elif dataset == "Spring":
        # The Spring dataset from L. Mehl et al. "Spring: A high-resolution high-detail dataset and benchmark for scene flow, optical flow and stereo" (CVPR 2023)
        if mode == 'training':
            dataset = datasets.Spring(split=Paths.splits("spring_train"), root=Paths.config("spring"), has_gt=True, fwd_only=False)
        elif mode == 'evaluation':
            dataset = datasets.Spring(split=Paths.splits("spring_eval"), root=Paths.config("spring"), has_gt=False, fwd_only=False)
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == "SpringSplitScheurer":
        # Spring-train and validation split from E. Scheurer et al. "Detection defenses: An empty promise against adversarial patch attacks on optical flow " (arXiv 2023)
        if mode == 'training':
            dataset = datasets.Spring(split=Paths.splits("spring_train"), root=Paths.config("spring"), has_gt=True, 
            scenes=["0001", "0004", "0005", "0006", "0007", "0008", "0009", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0020", "0021", "0022", "0023", "0024", "0025", "0027", "0030", "0033", "0036", "0037", "0038", "0039", "0041", "0043", "0044", "0047"], 
            fwd_only=True, camera=["left"], half_dimensions=True)
        elif mode == 'evaluation':
            dataset = datasets.Spring(split=Paths.splits("spring_train"), root=Paths.config("spring"), has_gt=True, 
            scenes=["0002","0010","0018","0026","0032","0045"], 
            fwd_only=True, camera=["left"], half_dimensions=True)
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == "HD1KSplitScheurer":
        # HD1K-train and validation split from E. Scheurer et al. "Detection defenses: An empty promise against adversarial patch attacks on optical flow " (arXiv 2023)
        if mode == 'training':
            scenes_tr = ['000000','000001','000002','000003','000004','000005','000006','000007','000008','000010','000011','000012','000014','000015','000016','000017','000020','000021','000022','000023','000024','000025','000026','000027','000028','000029','000030','000031','000033','000034','000035']
            dataset = datasets.HD1K(root=Paths.config("hd1k"), has_gt=True, scenes=scenes_tr, half_dimensions=True)
        elif mode == 'evaluation':
            scenes_val = ["000009", "000013", "000018", "000019", "000032"]
            dataset = datasets.HD1K(root=Paths.config("hd1k"), has_gt=True, scenes=scenes_val, half_dimensions=True)
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == "DrivingSample":
        # Driving sample from E. Scheurer et al. "Detection defenses: An empty promise against adversarial patch attacks on optical flow " (arXiv 2023)
        if mode == 'training':
            dataset = datasets.Driving(root=Paths.config("driving"), has_gt=True, 
                        dstype=[dstype], focallength=["15mm"], drivingcamview=["forward"], direction=["forward"], speed=["fast"], camera=["left"])
        elif mode == 'evaluation':
            dataset = datasets.Driving(root=Paths.config("driving"), has_gt=True, 
                        dstype=[dstype], focallength=["15mm"], drivingcamview=["forward"], direction=["forward"], speed=["fast"], camera=["left"])
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    else:
        raise ValueError("Unknown dataset %s, use either 'Sintel', 'Kitti15', 'Spring', 'SintelSplitZhao', 'SpringSplitScheurer', 'HD1KSplitScheurer' or 'DrivingSample'." %(dataset))
    # if e.g. the evaluation dataset does not provide a ground truth this is specified
    ds_has_gt = dataset.has_groundtruth()

    if small_run:
        reduced_num_samples = 32
        rand_indices = np.random.randint(0, len(dataset), reduced_num_samples)
        indices = np.arange(0, reduced_num_samples)
        dataset = Subset(dataset, indices)

    if num_repeats > 1:
        dataset = datasets.RepetitiveDataset(dataset, num_repeats=num_repeats)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), ds_has_gt


def preprocess_img(network, *images):
    """Manipulate input images, such that the specified network is able to handle them

    Args:
        network (str):
            Specify the network to which the input images are adapted

    Returns:
        InputPadder, *tensor:
            returns the Padder object used to adapt the image dimensions as well as the transformed images
    """
    if network in ['RAFT', 'GMA', 'FlowFormer']:
        padder = InputPadder(images[0].shape)
        output = padder.pad(*images)

    elif network == 'PWCNet':
        images = [(img / 255.) for img in images]
        padder = InputPadder(images[0].shape, divisor=64)
        output = padder.pad(*images)

    elif network == 'SpyNet':
        # normalize images to [0, 1]
        images = [ img / 255. for img in images ]
        # make image divisibile by 64
        padder = InputPadder(images[0].shape, divisor=64)
        output = padder.pad(*images)

    elif network[:7] == 'FlowNet':
        # normalization only for FlowNet, not FlowNet2
        if not network[:8] == 'FlowNet2':
            images = [ img / 255. for img in images ]
        # make image divisibile by 64
        padder = InputPadder(images[0].shape, divisor=64)
        output = padder.pad(*images)

    else:
        padder = None
        output = images

    return padder, output


def postprocess_flow(network, padder, *flows):
    """Manipulate the output flow by removing the padding

    Args:
        network (str): name of the network used to create the flow
        padder (InputPadder): instance of InputPadder class used during preprocessing
        flows (*tensor): (batch) of flow fields

    Returns:
        *tensor: output with removed padding
    """

    if padder != None:
        # remove padding
        return [padder.unpad(flow) for flow in flows]

    else:
        return flows


def compute_flow(model, network, x1, x2, test_mode=True, **kwargs):
    """subroutine to call the forward pass of the network

    Args:
        model (torch.nn.module):
            instance of optical flow model
        network (str):
            name of the network. [scaled_input_model | RAFT | GMA | FlowNet2 | SpyNet | PWCNet]
        x1 (tensor):
            first image of a frame sequence
        x2 (tensor):
            second image of a frame sequence
        test_mode (bool, optional):
            applies only to RAFT and GMA such that the forward call yields only the final flow field. Defaults to True.

    Returns:
        tensor: optical flow field
    """
    if network == "scaled_input_model":
        flow = model(x1,x2, test_mode=True, **kwargs)

    elif network == 'RAFT':
        _, flow = model(x1, x2, test_mode=test_mode, **kwargs)

    elif network == 'GMA':
        _, flow = model(x1, x2, iters=6, test_mode=test_mode, **kwargs)

    elif network == 'FlowFormer':
        flow = model(x1, x2)[0]

    elif network == 'FlowNetCRobust':
        flow = model(x1, x2)

    elif network[:7] == 'FlowNet':
        # all flow net types need image tensor of dimensions [batch, colors, image12, x, y] = [b,3,2,x,y]
        # FlowNet2-variants: all fine now, input [0,255] is taken.

        flow = model(x1,x2) # for FlowNetC/FlowNetC.py
        # x = torch.stack((x1, x2), dim=-3)
        # flow = model(x) # for FlowNet/FlowNetC.py

    elif network in ['PWCNet', 'SpyNet']:
        with warnings.catch_warnings():
            # this will catch the deprecated warning for spynet and pwcnet to avoid messy console
            warnings.filterwarnings("ignore", message="nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")
            warnings.filterwarnings("ignore", message="Default upsampling behavior when mode={} is changed")
            warnings.simplefilter("ignore", UserWarning)
            flow = model(x1,x2, **kwargs)

    else:
        flow = model(x1,x2, **kwargs)

    return flow


def model_takes_unit_input(model):
    """Boolean check if a network needs input in range [0,1] or [0,255]

    Args:
        model (str):
            name of the model

    Returns:
        bool: True -> [0,1], False -> [0,255]
    """
    if model in ["PWCNet", "SpyNet", "FlowNetCRobust"] or (model[:7] == 'FlowNet' and not model[:8]=='FlowNet2'):
        return True
    return False


def flow_length(flow):
    """Calculates the length of the flow vectors of a flow field

    Args:
        flow (tensor):
            flow field tensor of dimensions (b,2,H,W) or (2,H,W)

    Returns:
        torch.float: length of the flow vectors f_ij, computed as sqrt(u_ij^2 + v_ij^2) in a tensor of (b,1,H,W) or (1,H,W)
    """
    flow_pow = torch.pow(flow,2)
    flow_norm_pow = torch.sum(flow_pow, -3, keepdim=True)

    return torch.sqrt(flow_norm_pow)


def maximum_flow(flow):
    """Calculates the length of the longest flow vector of a flow field

    Args:
        flow (tensor):
            a flow field tensor of dimensions (b,2,H,W) or (2,H,W)

    Returns:
        float: length of the longest flow vector f_ij, computed as sqrt(u_ij^2 + v_ij^2)
    """
    return torch.max(flow_length(flow)).cpu().detach().numpy()


def quickvis_tensor(t, filename):
    """Saves a tensor with three dimensions as image to a specified file location.

    Args:
        t (tensor):
            3-dimensional tensor, following the dimension order (c,H,W)
        filename (str):
            name for the image to save, including path and file extension
    """
    # check if filename already contains .png extension
    if not filename.endswith('.png'):
        filename += '.png'
    valid = False
    if len(t.size())==3:
        img = t.detach().cpu().numpy()
        valid = True

    elif len(t.size())==4 and t.size()[0] == 1:
        img = t[0,:,:,:].detach().cpu().numpy()
        valid = True

    else:
        print("Encountered invalid tensor dimensions %s, abort printing." %str(t.size()))

    if valid:
        img = np.rollaxis(img, 0, 3)
        data = img.astype(np.uint8)
        data = Image.fromarray(data)
        data.save(filename)


def quickvisualization_tensor(t, filename, min=0., max=255.):
    """Saves a batch (>= 1) of image tensors with three dimensions as images to a specified file location.
    Also rescales the color values according to the specified range of the color scale.

    Args:
        t (tensor):
            batch of 3-dimensional tensor, following the dimension order (b,c,H,W)
        filename (str):
            name for the image to save, including path and file extension. Batches will append a number at the end of the filename.
        min (float, optional):
            minimum value of the color scale used by tensor. Defaults to 0.
        max (float, optional):
            maximum value of the color scale used by tensor Defaults to 255.
    """
    # rescale to [0,255]
    t = (t.detach().clone() - min) / (max - min) * 255.

    if len(t.size())==3 or (len(t.size())==4 and t.size()[0] == 1):
        quickvis_tensor(t, filename)

    elif len(t.size())==4:
        for i in range(t.size()[0]):
            if i == 0:
                quickvis_tensor(t[i,:,:,:], filename)
            else:
                quickvis_tensor(t[i,:,:,:], filename+"_"+str(i))

    else:
        print("Encountered unprocessable tensor dimensions %s, abort printing." %str(t.size()))


def quickvis_flow(flow, filename, auto_scale=True, max_scale=-1):
    """Saves a flow field tensor with two dimensions as image to a specified file location.

    Args:
        flow (tensor):
            2-dimensional tensor (c=2), following the dimension order (c,H,W) or (1,c,H,W)
        filename (str):
            name for the image to save, including path and file extension.
        auto_scale (bool, optional):
            automatically scale color values. Defaults to True.
        max_scale (int, optional):
            if auto_scale is false, scale flow by this value. Defaults to -1.
    """
    from flow_plot import colorplot_light
    # check if filename already contains .png extension
    if not filename.endswith('.png'):
        filename += '.png'
    valid = False
    if len(flow.size())==3:
        flow_img = flow.clone().detach().cpu().numpy()
        valid = True

    elif len(flow.size())==4 and flow.size()[0] == 1:
        flow_img = flow[0,:,:,:].clone().detach().cpu().numpy()
        valid = True

    else:
        print("Encountered invalid tensor dimensions %s, abort printing." %str(flow.size()))

    if valid:
        # make directory and ignore if it exists
        if not os.path.dirname(filename) == "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        # write flow
        flow_img = np.rollaxis(flow_img, 0, 3)
        data = colorplot_light(flow_img, auto_scale=auto_scale, max_scale=max_scale, return_max=False)
        data = data.astype(np.uint8)
        data = Image.fromarray(data)
        data.save(filename)


def quickvisualization_flow(flow, filename, auto_scale=True, max_scale=-1):
    """Saves a batch (>= 1) of 2-dimensional flow field tensors as images to a specified file location.

    Args:
        flow (tensor):
            single or batch of 2-dimensional flow tensors, following the dimension order (c,H,W) or (b,c,H,W)
        filename (str):
            name for the image to save, including path and file extension.
        auto_scale (bool, optional):
            automatically scale color values. Defaults to True.
        max_scale (int, optional):
            if auto_scale is false, scale flow by this value. Defaults to -1.
    """
    if len(flow.size())==3 or (len(flow.size())==4 and flow.size()[0] == 1):
        quickvis_flow(flow, filename, auto_scale=auto_scale, max_scale=max_scale)

    elif len(flow.size())==4:
        for i in range(flow.size()[0]):
            if i == 0:
                quickvis_flow(flow[i,:,:,:], filename, auto_scale=auto_scale, max_scale=max_scale)
            else:
                quickvis_flow(flow[i,:,:,:], filename+"_"+str(i), auto_scale=auto_scale, max_scale=max_scale)

    else:
        print("Encountered unprocessable tensor dimensions %s, abort printing." %str(flow.size()))


def show_images(*imgs, names=None, colorbars=False, wait=False, show=False, save=False, path='./plot.png', dpi=300):
    """plots images in a row in a single window. This function is only used for debugging purposes.
    Args:
        *imgs (tuple): Will be called with ax.imshow(imgs[i])
        colorbars (bool, optional): If True, all images will have a colorbar. Defaults to False.
        wait (bool, optional): If plot is not blocking will wait for buttonpress if True. Defaults to False.
        show (bool, optional): If True, the plot will block until window closed. Defaults to False.
    """
    imgs=list(imgs)
    # convert torch to numpy and squeeze batch dimension
    for i in range(len(imgs)):
        if isinstance(imgs[i], torch.Tensor):
            imgs[i] = imgs[i].detach().cpu().numpy().squeeze()
        if isinstance(imgs[i], np.ndarray):
            imgs[i] = imgs[i].squeeze()

    # if first dim is 2, assume it is a flow field
    for i in range(len(imgs)):
        if imgs[i].shape[0]==2:
            from flow_plot import colorplot_light
            imgs[i] = colorplot_light(imgs[i].transpose(1,2,0), return_max=False)
        elif imgs[i].shape[-1]==2:
            from flow_plot import colorplot_light
            imgs[i] = colorplot_light(imgs[i], return_max=False)
        if imgs[i].shape[0]==3:
            imgs[i] = imgs[i].transpose(1,2,0)

    n = len(imgs)
    fig, axes = plt.subplots(1, n, constrained_layout=True)
    if n == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        if imgs[i].shape[-1] == 3:
            p = ax.imshow(imgs[i])
        else:
            p = ax.imshow(imgs[i], cmap='gray')
        ax.axis('off')
        if names is not None:
            ax.set_title(names[i], fontsize='small', fontname='serif')
        if colorbars:
            plt.colorbar(p, ax=ax)
    if save:
        # if one image save with plt.imsave
        if n == 1:
            if imgs[0].shape[-1] == 3:
                plt.imsave(path, imgs[0])
            else:
                plt.imsave(path, imgs[0], cmap='gray')
            return
        plt.savefig(path, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    elif wait:
        plt.show(block=False)
        plt.pause(.1)
        plt.waitforbuttonpress()


def valid(X,i,j):
    """ check if the pixel is in the image """
    if i<0 or j<0 or i>=X.shape[-2] or j>=X.shape[-1]:
        return False
    return True


def B_d(I,i,j,e=5):
    """ return the list of valid pixels in the ball of radius e centered in (i,j) """
    res = []
    for di in range(-e,e+1):
        for dj in range(-e,e+1):
            if valid(I,i+di,j+dj) and di**2 + dj**2 <= e**2 and (di!=0 or dj!=0):
                res.append((i+di,j+dj))
    return res







