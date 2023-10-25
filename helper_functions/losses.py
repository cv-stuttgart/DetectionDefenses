import torch
import torch.nn.functional as tnf


def aae_masked(F,F_def,M=None):
    """Computes the average angular error (AAE) between two flow fields.
    The mask is used to ignore the pixels where a patch is present.

    Args:
        F (tensor):
            predicted flow field
        F_def (tensor):
            target flow field
        M (tensor, optional):
            mask of the patch. Defaults to None.
    
    Returns:
        float: scalar AAE value
    """
    N,C,H,W=F.size()
    pad = torch.ones((N,1,H,W)).to(F.device)
    cs =tnf.cosine_similarity(torch.cat((F,pad),dim=1),torch.cat((F_def,pad),dim=1))
    return (cs*M).sum()/(M.sum()) if M is not None else cs.sum()/cs.numel()
    

def acs_masked(F,F_def,M=None):
    """
    Computes the average cosine similarity (ACS) between two flow fields.
    The mask is used to ignore the pixels where a patch is present.

    Args:
        F (tensor):
            predicted flow field
        F_def (tensor):
            target flow field
        M (tensor, optional):
            mask of the patch. Defaults to None.
    
    Returns:
        float: scalar ACS value
    """
    cs = tnf.cosine_similarity(F,F_def)
    return (cs*M).sum()/(M.sum()) if M is not None else cs.sum()/cs.numel()


def aee_masked(F,F_def,M=None):
    """
    Computes the average end point error (AEE) between two flow fields.
    The mask is used to ignore the pixels where a patch is present.

    Args:
        F (tensor):
            predicted flow field
        F_def (tensor):
            target flow field
        M (tensor, optional):
            mask of the patch. Defaults to None.
    
    Returns:
        float: scalar AEE value
    """
    u,v = F[:,0,:,:],F[:,1,:,:]
    u_def,v_def = F_def[:,0,:,:],F_def[:,1,:,:]
    se = (u-u_def)**2+(v-v_def)**2
    se[se == 0.0] = 1e-10
    ee = torch.sqrt(se)
    return (ee*M).sum()/(M.sum()) if M is not None else ee.sum()/ee.numel()


def mse_masked(F,F_def,M=None):
    """
    Computes the mean squared error (MSE) between two flow fields.
    The mask is used to ignore the pixels where a patch is present.

    Args:
        F (tensor):
            predicted flow field
        F_def (tensor):
            target flow field
        M (tensor, optional):
            mask of the patch. Defaults to None.
    
    Returns:
        float: scalar MSE value
    """
    u,v = F[:,0,:,:],F[:,1,:,:]
    u_def,v_def = F_def[:,0,:,:],F_def[:,1,:,:]
    se = ((u-u_def)**2+(v-v_def)**2)
    return (se*M).sum()/(M.sum()) if M is not None else se.sum()/se.numel()
    