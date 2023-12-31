from torch.nn.modules.module import Module
from torch.autograd import Function, Variable


import os
import sys
# You might have to add the path where resample2d_cuda-0.0.0-py3.6-linux-x86_64.egg can be found after it was compiled.
# Very likely something like /lib/pythonX.X/site-packages/
# sys.path.append("resample2d_cuda-0.0.0-py3.6-linux-x86_64.egg")
#sys.path.append("$HOME/.local/lib/python3.6/site-packages/resample2d_cuda-0.0.0-py3.6-linux-x86_64.egg")
import resample2d_cuda

class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1, bilinear= True):
        assert input1.is_contiguous()
        assert input2.is_contiguous()

        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        ctx.bilinear = bilinear

        _, d, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.new(b, d, h, w).zero_()

        resample2d_cuda.forward(input1, input2, output, kernel_size, bilinear)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()

        input1, input2 = ctx.saved_tensors

        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())

        resample2d_cuda.backward(input1, input2, grad_output.data,
                                 grad_input1.data, grad_input2.data,
                                 ctx.kernel_size, ctx.bilinear)

        return grad_input1, grad_input2, None, None

class Resample2d(Module):

    def __init__(self, kernel_size=1, bilinear = True):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.bilinear = bilinear

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size, self.bilinear)
