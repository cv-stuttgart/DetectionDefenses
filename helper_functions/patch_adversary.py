import torch
import os
import torchvision.transforms.functional as tvf
from torch.nn.functional import pad
from numpy import load as load_npy


"""
A module representing an adversarial  patch adversary that performs input manipulations 

Inputs:
P (init) : None, Filepath to png or tensor
I1,I2    : Input image pair to attack as tensors of size (N x 3 x H x W)
y,x      : position of the patch center in image coordinates
angle    : a list [min, max] or a float specifying the rotation angle range
sx,sy    : lists[min,max] or float specifying the vertical/horizontal scaling factor

Especially considered sources:
https://discuss.pytorch.org/t/how-to-make-a-tensor-part-of-model-parameters/51037/6
"""


def circ_mask(X, /, k=0):
    """Creates a circular mask of size X.shape[-2:] with radius k"""
    from math import sqrt
    N, C, H, W = X.size()
    R = torch.zeros((N, 1, H, W))
    for i in range(-H//2, H//2):
        for j in range(-W//2, W//2):
            R[:, :, i+H//2, j+W //
                2] = 1 if sqrt((i+0.5)**2+(j+0.5)**2) < H//2-k else 0
    return R


class PatchAdversary(torch.nn.Module):
    def __init__(self, P, *, angle=[-10, 10], scale=[0.95, 1.05], size=None, change_of_variable=False):
        """Adversarial patch that when images passed through applies the patch to the image. This class is used for the patch attack with defense.

        Args:
            P (str): path to png file or numpy array. If None, a zero patch is initialized.
            angle (list): [min,max] or float specifying the rotation angle range
            scale (list): [min,max] or float specifying the vertical/horizontal scaling factor
            change_of_variable (bool, optional): If True, a change of variable is used to ensure patch+image is in the range [0,1]. Defaults to False.
        """
        super(PatchAdversary, self).__init__()

        # Initialize Patch from random, filepath or tensor
        if P is None:
            assert size is not None, "If no patch is given, a size must be specified"
            self.P = torch.zeros((1, 3, size, size))
            # P=torch.rand((1,3,size,size))
            self.M = circ_mask(self.P)
        elif isinstance(P, str):
            if P.endswith('.png'):
                from PIL import Image
                P = tvf.to_tensor(Image.open(P)).unsqueeze_(0)
            elif P.endswith('.npy'):
                P = torch.from_numpy(load_npy(P))

            if P.size(1) == 4:  # compatibility with old version
                self.P = P[:, :3]
                self.M = P[:, 3:]
            else:
                raise ValueError("Patch must have 4 channels (RGB and alpha)")
            
        elif isinstance(P, torch.Tensor):
            if P.size(1) == 4:
                self.P = P[:, :3]
                self.M = P[:, 3:]
            else:
                raise ValueError("Patch must have 4 channels (RGB and alpha)")

        self.P = torch.nn.Parameter(self.P, requires_grad=True)
        self.M = torch.nn.Parameter(self.M, requires_grad=False)
        self.angle = angle
        self.scale = scale

        self.cov = change_of_variable

    def get_P(self, Mask=False):
        if self.cov:
            P = .5*(torch.tanh(self.P)+1)
        else:
            P = self.P
        if Mask:
            return torch.concat([P, self.M], dim=1)
        else:
            return P

    def forward(self, I1, I2, y=None, x=None):

        scale = self.scale
        angle = self.angle

        # Initialize Transformations
        scale = torch.rand((1,)).item()*(scale[1]-scale[0])+scale[0] \
            if isinstance(scale, list) else scale

        angle = torch.rand((1,)).item()*(angle[1]-angle[0])+angle[0] \
            if isinstance(angle, list) else angle

        # Apply Transformations
        if angle != 0 and scale != 1:
            P_rot = tvf.rotate(self.P, angle)
            P_res = tvf.resize(
                P_rot, (int(scale*P_rot.size(2)), int(scale*P_rot.size(3))))
            M = tvf.rotate(self.M, angle)
            M = tvf.resize(M, (int(scale*M.size(2)), int(scale*M.size(3))))
        else:
            P_res = self.P
            M = self.M

        # Initialize pos, patch and patch size
        N, C, H, W = I1.size()
        n, c, h, w = P_res.size()
        assert H >= h and W >= w, "Patch size must be smaller than image size"

        # random loc
        y = torch.randint(H-h, (1,)).item()+(h//2) if y is None else y
        x = torch.randint(W-w, (1,)).item()+(w//2) if x is None else x

        # Resize patch to size of one image
        P_glob = pad(P_res, ((x-w//2), W-w-(x-w//2), (y-h//2), H-h-(y-h//2)))
        M_glob = pad(M, ((x-w//2), W-w-(x-w//2), (y-h//2), H-h-(y-h//2)))

        if self.cov:
            P_glob_cov = .5*(torch.tanh(P_glob)+1)
        else:
            P_glob_cov = P_glob

        # ceil patch to avoid black borders
        M_glob = torch.ceil(M_glob)

        # Construct result (replace image where patch is not transparent)
        R1 = (1-M_glob)*I1 + P_glob_cov*M_glob
        R2 = (1-M_glob)*I2 + P_glob_cov*M_glob

        M = torch.where(M_glob > 0, 1, 0).to(I1.device)

        return R1, R2, M, y, x

    def save_png(self, name):
        if os.path.dirname(name) and not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))
        tvf.to_pil_image(torch.clip(self.get_P(Mask=True), 0, 1)[0]).save(name)
