import torch
import sys
from helper_functions.patch_adversary import PatchAdversary, circ_mask


def get_manual_patch(size=100):
    """Create patch that has the highest possible gradient for the given size."""
    # P = torch.rand((1, 3, size, size))
    # P = torch.round(P)

    # checkerboard
    P = torch.zeros((1, 3, size, size))
    for i in range(0, size):
        for j in range(size):
            if (i+j) % 2 == 0:
                P[0, :, i, j] = 1.

    P = torch.cat([P, circ_mask(P)], dim=1)
    P = PatchAdversary(P)
    return P


P = get_manual_patch(100)

# save patch to file
P.save_png('results/Manual/best-patches/patch.png')
