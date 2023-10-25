# Derived from the SGD optimizer implementation
# https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html

import torch

# based on work by Kurakin et al. https://arxiv.org/pdf/1611.01236.pdf
class IFGSM(torch.optim.Optimizer):
    def __init__(self,params,lr,min_,max_):
        defaults = dict(lr=lr,momentum=0,dampending=0,weight_decay=0,\
                        nesterov=False,maximize=False)
        super(IFGSM,self).__init__(params,defaults)
        self.min_=min_
        self.max_=max_

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # p.grad[:,3,:,:]=0
                p.copy_(torch.clip(p-group["lr"]*p.grad.sign(),self.min_,self.max_))

        return loss
           
# based on the descriptions of Carlini et al. https://arxiv.org/pdf/1608.04644.pdf
# and the update in the flowattack repository by Ranjan et al.
# https://github.com/anuragranj/flowattack
class ClippedPGD(torch.optim.Optimizer):
    def __init__(self,params,lr,min_,max_,max_delta):
        defaults = dict(lr=lr,momentum=0,dampending=0,weight_decay=0,\
                        nesterov=False,maximize=False)
        super(ClippedPGD,self).__init__(params,defaults)
        self.min_ = min_
        self.max_ = max_
        self.max_delta=max_delta
        
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                delta=torch.clip(group["lr"]*p.grad,-self.max_delta,self.max_delta)
                p.copy_(torch.clip(p-delta,self.min_,self.max_))
        return loss
