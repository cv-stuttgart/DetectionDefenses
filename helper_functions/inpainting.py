# opencv for inpainting and numpy to convert pytorch tensors
# to a compatible format for opencv
import cv2
import numpy as np
import torch

# Implementation of priority queue in our inpainting algorithm
# based on the inpainting algorithm published by telea et al.
import heapq

# Tensor transformations
import torch.nn.functional as tnf

from helper_functions.ownutilities import B_d, valid

# Custom imports

# This implementation considers the following sources
# very central is the pseudocode of the algorithm by Telea
# "An image inpainting technique based on the fast marching method"
# we accessed it at:
# https://www.researchgate.net/publication/238183352_An_Image_Inpainting_Technique_Based_on_the_Fast_Marching_Method
#
# additionally this code considered existing implementations of 
# pyheal: https://github.com/olvb/pyheal
# python-opencv: https://github.com/opencv/opencv-python
# telea2004: https://github.com/erich666/jgt-code/tree/master/Volume_09/Number_1/Telea2004

EPS=1e-6
INF=1e6
KNOWN = 0
BAND = 1
INSIDE = 2


##############################################################
# Approach for direct reimplementation
##############################################################

def fmm(I,M,d=5,T_max=INF):
    """
    Input:
    I : tensor of input image of dimension 3 x H x W
    M : tensor of binary mask of dimension 1 x H x W
    
    L    : Field of labels
    T    : Field of level-set distances
    band : Heap of band pixels ordered by distance (dist, (y,x))
    """
    
    # This function is overloaded.
    # It performs Teleas inpainting if I is not None and returns I_inp
    # Else it computes the level set distances of the marked regions
    if I is not None:
        # T values for outer pixels
        T_out=fmm(None,(1-tnf.max_pool2d(M,3,padding=1,stride=1)),d,d).double()

    
    # L 2 for inner 1 for border and 0 for outer
    L = (M+tnf.max_pool2d(M.clone(),3,padding=1,stride=1)).squeeze_(0).squeeze_(0)
    T = T_out if I is not None else (torch.abs(L-1)*INF)
    band = [(0,(idx[0].item(),idx[1].item())) for idx in torch.nonzero(L == BAND)]
    heapq.heapify(band)

    while band:
        t,(i,j) = heapq.heappop(band)
        L[i,j] = 0
        for k,l in [(i-1,j),(i,j-1),(i+1,j),(i,j+1)]:
            if valid(L,k,l) and L[k,l] != KNOWN:
                if L[k,l] == INSIDE:
                    L[k,l] = BAND
                    
                    
                    M_ = torch.zeros_like(M)
                    M_[:,k,l] = 1
                    if I is not None:
                        I = (1-M_)*I+M_*(M[:,k,l]*inp(I,T,L,k,l,d)+(1-M[:,k,l])*I[:,k,l]).unsqueeze(1).unsqueeze(1)
                t_new = min([solve(L,T,a,b,c,d) for a,b,c,d in [[k-1,l,k,l-1],
                                                                [k+1,l,k,l-1],
                                                                [k-1,l,k,l+1],
                                                                [k+1,l,k,l+1]]])
                if T[k,l] > t_new and t_new<T_max:
                    heapq.heappush(band,(t_new,(k,l)))
                    T[k,l] = t_new
                    
    return I if I is not None else T


def solve(L,T,i1,j1,i2,j2):
    
    if not valid(L,i1,j1) or not valid (L,i2,j2):
        return 10e6
    
    if L[i1,j1] == KNOWN:
        if L[i2,j2] == KNOWN:
            if 2-(T[i1,j1]-T[i2,j2])**2 >0:
                r = torch.sqrt(2-(T[i1,j1]-T[i2,j2])**2)
                s = (T[i1,j1]+T[i2,j2]-r)/2
                if s >= T[i1,j1] and s >= T[i2,j2]:
                    return s
                else:
                    s += r
                    if s >= T[i1,j1] and s >= T[i2,j2]:
                        return s
        else:
            return 1 + T[i1,j1]
    elif L[i2,j2] == KNOWN:
        return 1 + T[i2,j2]
    return 1.0e6


#################################################
# Simplified approaches
#################################################
def telea(I,M,d=5):
    """
    Input:
    I : tensor of input image of dimension 3 x H x W
    M : tensor of binary mask of dimension 1 x H x W
    
    
    """
    
    # L    : Field of labels
    # T    : Field of level-set distances
    # band : Heap of band pixels ordered by distance (dist, (y,x))
    L = (M.clone()+tnf.max_pool2d(M.clone(),3,padding=1,stride=1)).squeeze_(0).detach() 
    T = (torch.abs(L-1)*10e6).detach()
    band = [(0,(idx[0].item(),idx[1].item())) for idx in torch.nonzero(L == BAND)]
    heapq.heapify(band)


    inpaintings = []
    while band:
        t,(i,j) = heapq.heappop(band)
        L[i,j] = 0
        for k,l in [(i-1,j),(i,j-1),(i+1,j),(i,j+1)]:
            if valid(L,k,l) and L[k,l] != KNOWN:
                if L[k,l] == INSIDE:
                    L[k,l] = BAND
                    
                    #
                    M_ = torch.zeros_like(M)
                    M_[:,k,l] = 1
                    I = (1-M_)*I+M_*(M[:,k,l]*inp(I,T,L,k,l,d)+(1-M[:,k,l])*I[:,k,l]).unsqueeze(1).unsqueeze(1)
                t_new = min([solve(L,T,a,b,c,d) for a,b,c,d in [[k-1,l,k,l-1],
                                                                [k+1,l,k,l-1],
                                                                [k-1,l,k,l+1],
                                                                [k+1,l,k,l+1]]])
                if T[k,l] > t_new:
                    heapq.heappush(band,(t_new,(k,l)))
                    T[k,l] = t_new
                    
    return I
    

def inp(I,T,L,i,j,d):
    Ia = torch.zeros((3),requires_grad=I.requires_grad)
    s  = torch.zeros((1),requires_grad=I.requires_grad)
    (Ia,s) = (Ia.to(I.device),s.to(I.device)) if I.is_cuda else (Ia,s)
    for k,l in [(k,l) for k,l in B_d(I,i,j,d) if L[k,l] != INSIDE]:
        
        Ia = Ia + I[:,k,l]
        s= s + torch.tensor(1.,requires_grad=I.requires_grad)
    return Ia/s
