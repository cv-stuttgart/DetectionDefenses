#from attrdict import AttrDict
import torch
import cv2
import torch.nn.functional as tnf
import torch.nn.functional as F
import numpy as np

from helper_functions.inpainting import fmm, telea

#
# The defenses coded here are based on their original publications
# Naseer et al. for LGS : https://arxiv.org/pdf/1807.01216.pdf
# Anand et al. for ILP  : https://ieeexplore.ieee.org/abstract/document/9356338
#
# The implementations are significantly influenced by an (inofficial) implementation of Local gradients smoothing
# https://github.com/metallurk/local_gradients_smoothing
#
#



####
#  Kernels
####
central_h=torch.tensor([[[[ 0.,0.,0.],[-1./2,0.,1./2],[0.,0.,0.]]]])
central_v = central_h.permute(0,1,3,2).contiguous()

sobel_h = torch.tensor([[[[1./8,0.,-1./8],[2./8,0.,-2./8],[1./8,0.,-1./8]]]])
sobel_v = sobel_h.permute(0,1,3,2).contiguous()

forward_h = torch.tensor([[[[0.,0.,0.],[0.,-1.,1.],[0.,0.,0.]]]])
forward_v = forward_h.permute(0,1,3,2).contiguous()

laplacian = torch.tensor([[[[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]]]])

tent_kernel = torch.tensor([[[[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]]]])

kernels = {"forward"        : [forward_v,forward_h],
           "central_diff"   : [central_v,central_h],
           "sobel"          : [sobel_v,sobel_h],
           "laplacian"      : [laplacian],
           "tent"           : [tent_kernel]
          }


######################################
# Defenses for Optical Flow Networks
######################################
class Defense(torch.nn.Module):
    def __init__(self,defense_name):
        super(Defense,self).__init__()
        self.defense_name = defense_name

    def forward(self, I1, I2, M=None):
        raise NotImplementedError("Defense.forward() is not implemented")

class LGS(Defense):
    
    def __init__(self,k,o,t,s,discr):
        super().__init__("LGS")
        self.k=k
        self.o=o
        self.t=t
        self.s=s
        self.discr=discr        
    
    def forward(self,I1_batch,I2_batch,M_batch=None):
        Defended1, Defended2 = torch.zeros_like(I1_batch), torch.zeros_like(I2_batch)

        for i, (I1, I2) in enumerate(zip(I1_batch,I2_batch)):
            I1 = I1.unsqueeze(0)
            I2 = I2.unsqueeze(0)    

            # 1.) Joint Gradient Magnitude Field
            G1 = JointGradMag()(I1.clone(),alg=self.discr)
            G2 = JointGradMag()(I2.clone(),alg=self.discr)
            
            # 2.) Normalization
            min_1,max_1 = G1.min(),G1.max()
            min_2,max_2 = G2.min(),G2.max()
            G1norm = (G1-min_1)/(max_1-min_1) 
            G2norm = (G2-min_2)/(max_2-min_2)

            # 3.) Blockwise Filtering
            F1 = BlockwiseFilter()(G1norm,self.k,self.o,self.t)
            F2 = BlockwiseFilter()(G2norm,self.k,self.o,self.t)

            
            # 4.) Gradient Smoothing
            #R1 = I1*(1-BPDAClip.apply(self.s*F1))
            #R2 = I2*(1-BPDAClip.apply(self.s*F2))
            Defended1[i] = I1*(1-torch.clip(self.s*F1,0,1))
            Defended2[i] = I2*(1-torch.clip(self.s*F2,0,1))

        return Defended1, Defended2
    
class ILP(Defense):
    
    def __init__(self,k,o,t,s,r,discr,mode="bpda"):
        """Defense proposed by Anand et al. (2020). Detects patches by their local second order gradient magnitude and inpaints them.

        Args:
            k (int): blocksize
            o (int): overlap
            t (float): blockwise filtering threshold
            s (float): smoothing parameter
            r (int): inpainting radius
            discr (_type_): How the gradient is computed. eg "forward" or "central_diff"
            mode (str, optional): Inpainting mode. Defaults to "bpda".
        """
        super().__init__("ILP")
        self.k=k
        self.o=o
        self.t=t
        self.s=s
        self.r=r
        self.discr=discr
        self.mode=mode
    
    def forward(self,I1_batch,I2_batch,M=None):
        Defended1, Defended2 = torch.zeros_like(I1_batch), torch.zeros_like(I2_batch)

        for i, (I1, I2) in enumerate(zip(I1_batch,I2_batch)):
            I1 = I1.unsqueeze(0)
            I2 = I2.unsqueeze(0)

            # 1.) Second order gradient magnitude field
            G1 = Joint2ndGradMag()(I1.clone(),alg=self.discr)
            G2 = Joint2ndGradMag()(I2.clone(),alg=self.discr)

            # 2.) Normalization
            min_1,max_1 = G1.min(),G1.max()
            min_2,max_2 = G2.min(),G2.max()
            G1norm = (G1-min_1)/(max_1-min_1) 
            G2norm = (G2-min_2)/(max_2-min_2) 
            
            
            # 3.) Blockwise Filtering
            G1_filt = BlockwiseFilter()(G1norm.clone(),self.k,self.o,self.t)
            G2_filt = BlockwiseFilter()(G2norm.clone(),self.k,self.o,self.t)

            
            # 4.) Pixelwise Filtering
            M1 = BPDAGT.apply(self.s*G1_filt,0.5,1.0,0.0)
            M2 = BPDAGT.apply(self.s*G2_filt,0.5,1.0,0.0)

        
            # 5.) Morphological Closing
            M1_closed = BPDAClosing.apply(M1)
            M2_closed = BPDAClosing.apply(M2)
        
            # 6.) Inpainting
            # In the final submission we only used self.mode == "bpda"
            if self.mode == "simplified":
                # Mask indicating region to inpaint around patch with autograd
                M_inner, y_min,y_max,x_min,x_max = mask_and_bounds(M,5)
            
                # Inpainting patch section of the image (autograd)
                I1_autograd_unpadded = telea(I1[0,:,y_min:y_max,x_min:x_max].clone(),\
                                        M1_closed[0,:,y_min:y_max,x_min:x_max].clone(),5)
                I2_autograd_unpadded = telea(I2[0,:,y_min:y_max,x_min:x_max].clone(),\
                                        M2_closed[0,:,y_min:y_max,x_min:x_max].clone(),5)        
                I1_autograd=tnf.pad(I1_autograd_unpadded,(x_min,I1.size(3)-x_max,y_min,I1.size(2)-y_max))
                I2_autograd=tnf.pad(I2_autograd_unpadded,(x_min,I2.size(3)-x_max,y_min,I2.size(2)-y_max))

                I1_cv = BPDAInpaintingCV.apply(I1,M1_closed,5)
                I2_cv = BPDAInpaintingCV.apply(I2,M2_closed,5)

                # Combine Results
                Defended1[i] = M_inner*I1_autograd+(1-M_inner)*I1_cv
                Defended2[i] = M_inner*I2_autograd+(1-M_inner)*I2_cv
            elif self.mode=="full":
                # Mask indicating region to inpaint around patch with autograd
                M_inner, y_min,y_max,x_min,x_max = mask_and_bounds(M,5)
            
                # Inpainting patch section of the image (autograd)
                I1_autograd_unpadded = fmm(I1[0,:,y_min:y_max,x_min:x_max].clone(),\
                                        M1_closed[0,:,y_min:y_max,x_min:x_max].clone(),5)
                I2_autograd_unpadded = fmm(I2[0,:,y_min:y_max,x_min:x_max].clone(),\
                                        M2_closed[0,:,y_min:y_max,x_min:x_max].clone(),5)        
                I1_autograd=tnf.pad(I1_autograd_unpadded,(x_min,I1.size(3)-x_max,y_min,I1.size(2)-y_max))
                I2_autograd=tnf.pad(I2_autograd_unpadded,(x_min,I2.size(3)-x_max,y_min,I2.size(2)-y_max))

                I1_cv = BPDAInpaintingCV.apply(I1,M1_closed,5)
                I2_cv = BPDAInpaintingCV.apply(I2,M2_closed,5)

                # Combine Results
                Defended1[i] = M_inner*I1_autograd+(1-M_inner)*I1_cv
                Defended2[i] = M_inner*I2_autograd+(1-M_inner)*I2_cv
            elif self.mode=="bpda":
                Defended1[i] = BPDAInpaintingCV.apply(I1,M1_closed,self.r)
                Defended2[i] = BPDAInpaintingCV.apply(I2,M2_closed,self.r)
        
        return Defended1, Defended2

###############################
# Gradient Field Computation
###############################

class JointGradMag(torch.nn.Module):
    
    def forward(self,I,alg):
        
        N,C,H,W = I.size()
        kv,kh = kernels[alg]
        kv,kh = kv.to(I.device),kh.to(I.device)
        
        # Perform Convolution
        I_pad = tnf.pad(I,(1,1,1,1),mode="reflect")
        dy = tnf.conv2d(I_pad.view(-1,1,H+2,W+2).contiguous(),kv,padding="valid").view(N,C,H,W).contiguous()
        dx = tnf.conv2d(I_pad.view(-1,1,H+2,W+2).contiguous(),kh,padding="valid").view(N,C,H,W).contiguous()
        
        # Joint Magnitude
        G_joint_mag=torch.sqrt(torch.sum(dx**2,1,keepdim=True)+\
                               torch.sum(dy**2,1,keepdim=True)+1e-6)
        
        return G_joint_mag        
    
    
class Joint2ndGradMag(torch.nn.Module):
    
    def forward(self,I,alg):
        
        N,C,H,W = I.size()
        (kv,kh) = kernels[alg]
        kv,kh = kv.to(I.device),kh.to(I.device)
        
        # Perform Convolution
        I_pad = tnf.pad(I,(1,1,1,1),mode="reflect")
        dy = tnf.conv2d(I_pad.view(-1,1,H+2,W+2).contiguous(),kv,padding="valid")\
                             .view(N,C,H,W).contiguous()
        dx = tnf.conv2d(I_pad.view(-1,1,H+2,W+2).contiguous(),kh,padding="valid")\
                             .view(N,C,H,W).contiguous()
        dy_pad = tnf.pad(dy,(1,1,1,1),mode="reflect")
        dx_pad = tnf.pad(dx,(1,1,1,1),mode="reflect")
        dydy=tnf.conv2d(dy_pad.view(-1,1,H+2,W+2).contiguous(),kv,padding="valid")\
                              .view(N,C,H,W).contiguous()
        dxdx=tnf.conv2d(dx_pad.view(-1,1,H+2,W+2).contiguous(),kh,padding="valid")\
                              .view(N,C,H,W).contiguous()

        
        # Joint Magnitude
        G_joint_mag=torch.sqrt(torch.sum(dxdx**2,1,keepdim=True)+\
                               torch.sum(dydy**2,1,keepdim=True)+1e-6)
        
        return G_joint_mag
    
    
###################################################
# Utility Functions for the optical flow defenses
###################################################

#https://discuss.pytorch.org/t/how-to-split-tensors-with-overlap-and-then-reconstruct-the-original-tensor/70261
# This function is especially inspired by the inofficial local gradients smoothing implementation
# https://github.com/metallurk/local_gradients_smoothing
class BlockwiseFilter(torch.nn.Module):
    
    def forward(self,G,k,o,t):
        
        # 0.) Initialize variables
        t = torch.rand(1)*(t[1]-t[0])+t[0] if isinstance(t,list) else t

        N,C,H,W = G.size()
        s = (k-o) 
        v_pad = o - (H % s) if H % s <= o else k - (H%s)
        h_pad = o - (W % s) if W % s <= o else k - (W%s)
        H_pad, W_pad = H + v_pad, W + h_pad
    
        # 1.) Padding
        padding=(int(h_pad/2),int(h_pad-int(h_pad/2)),int(v_pad/2),int(v_pad-int(v_pad/2)))
        G_pad = tnf.pad(G,padding,mode="reflect")
        
        # 2.) Block decomposition in tensor 1 x (k*k) x n
        B = tnf.unfold(G_pad,kernel_size=k,stride=k-o)
        _,n_b,l = B.size()
        
        # 3.) Statistic computation
        B_filt = B.sum(axis=1,keepdim=True)/(k*k)
        #M_B = BPDAWhere.apply(B_filt,t,1.0,0.0).expand(1,n_b,l).contiguous()
        M_B = torch.where(B_filt>t,1.0,0.0).expand(1,n_b,l).contiguous()
        
        # 3.) Compute averages        
        g_hat = tnf.fold(B*M_B,output_size=(H_pad,W_pad),kernel_size=k,stride=s)
        g_hat = BPDADiv.apply(g_hat,tnf.fold(torch.ones_like(B_filt)*M_B,output_size=(H_pad,W_pad),kernel_size=k,stride=s))
        g_hat = BPDAReplaceNaN.apply(g_hat)
        #g_hat[g_hat!=g_hat] = 0
                
        return tnf.pad(g_hat,tuple(-p for p in padding))

class BPDADiv(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,a,b):
        return a/b
    
    @staticmethod
    def backward(ctx,dT):
        return dT,dT    


class BPDAReplaceNaN(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,g_hat):
        g_hat[g_hat!=g_hat] = 0
        return g_hat
    
    @staticmethod
    def backward(ctx,dT):
        return dT    


def mask_and_bounds(M,d):
    N,C,H,W = M.size()
    inds = torch.nonzero(M)
    y_min = max(0,torch.min(inds[:,2]).item()-d)
    y_max = min(torch.max(inds[:,2]).item()+1+d,H)
    x_min = max(0,torch.min(inds[:,3]).item()-d)
    x_max = min(torch.max(inds[:,3]).item()+1+d,W)
    
    M_inner = torch.zeros_like(M)
    M_inner[:,:,y_min:y_max,x_min:x_max] = 1
    
    return M_inner,y_min,y_max,x_min,x_max    
    
    
    
    
#################################################
# Components Implementing BPDA
#################################################

# bpda Greater Than
class BPDAGT(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,t1,t,T1,T2):
        ctx.save_for_backward(t1,torch.tensor(t))
        return torch.where(t1>t,T1,T2)
    
    @staticmethod
    def backward(ctx,dT):
        t1,t = ctx.saved_tensors
        #dT_filt = torch.where(t1>t.float().cuda(),torch.tensor(1.0).cuda(),torch.tensor(0.0).cuda())
        #return dT_filt,None,None,None  
        return dT,None,None,None
    
class BPDAClosing(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,M):
        ctx.save_for_backward(M)
        M_dilated = tnf.max_pool2d(M,3,padding=3//2,stride=1)
        M_eroded = -tnf.max_pool2d(-M_dilated,3,padding=3//2,stride=1)
        return M_eroded
    
    @staticmethod
    def backward(ctx,dT):
        t1 = ctx.saved_tensors
        return None  
    
class BPDAInpaintingCV(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,I,M,d):
        I_cv=(255*I.cpu()[0].permute(1,2,0).clone().detach().numpy()).astype(np.uint8)
        M_cv=M.cpu()[0,0].clone().detach().numpy().astype(np.uint8)
        R_cv=cv2.inpaint(I_cv,M_cv,d,cv2.INPAINT_TELEA)
        R = (torch.from_numpy(R_cv)/255).permute(2,0,1).unsqueeze_(0).to(I.device).contiguous()
        ctx.save_for_backward(M)
        return R

    @staticmethod
    def backward(ctx,dR1):
        M = ctx.saved_tensors[0]
        dR1 = torch.where(M==0.0,dR1,torch.tensor(0.0).to(dR1.device))
        return dR1,None,None
