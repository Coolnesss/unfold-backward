import torch
import unfold
import numpy as np
import time

class UnfoldFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, size, step):
        ctx.input_sizes = input.shape
        ctx.dim = dim
        ctx.size = size
        ctx.step = step
        return input.unfold(dim, size, step)

    @staticmethod
    def backward(ctx, grad_input):
        input_sizes, dim, size, step = ctx.input_sizes, ctx.dim, ctx.size, ctx.step
        return unfold.backward(grad_input, input_sizes, dim, size, step), None, None, None


def test():
    dim=0
    for step in range(1,10):
        for size in range(10):
            for nelemts in [10,13,21,53,95]:

                a = torch.arange(1, nelemts+1).float()
                a.unfold(dim,size,step)
                a.requires_grad_()
                
                ret = UnfoldFunction.apply(a, dim, size, step)

                (ret).sum().backward()
                
                # Real 
                b = torch.arange(1,nelemts+1).float()
                b.requires_grad_()
                ret = b.unfold(dim, size, step)

                (ret).sum().backward()
                
                assert torch.allclose(b.grad, a.grad)

def main():
    # Mine
    dim = 0
    size = 3
    step = 1
    nelemts = 5*3

    a = torch.arange(1, nelemts+1).float().view(5, 3)
    print("INPUT:")
    print(a)
    print("OUTPUT:")
    print(a.unfold(dim,size,step))
    a.requires_grad_()
    
    ret = UnfoldFunction.apply(a, dim, size, step)

    (ret).sum().backward()
    print("mine", a.grad)
    
    # Real 
    b = torch.arange(1,nelemts+1).float().view(5, 3)
    b.requires_grad_()
    ret = b.unfold(dim, size, step)

    (ret).sum().backward()
    print("real",b.grad)
    
    assert torch.allclose(b.grad, a.grad)

def bench():

    nelem = 1e7
    dim = 0
    size = 3
    step = 1

    a = torch.arange(nelem).float()
    print(a.unfold(dim,size,step))
    a.requires_grad_()
    ret = UnfoldFunction.apply(a, dim, size, step)

    start = time.time()
    (ret).sum().backward()
    end = time.time()
    print("Took", end-start)

    # Real 
    b = torch.arange(nelem).float()
    b.requires_grad_()
    ret = b.unfold(dim, size, step)
    start = time.time()
    (ret).sum().backward()
    
    end = time.time()
    print("Took", end-start)
    
    assert torch.allclose(b.grad, a.grad)

if __name__ == '__main__':
    bench()