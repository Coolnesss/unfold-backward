import torch
import unfold
import numpy as np
import time
from random import shuffle

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
        if grad_input.is_cuda:
            return unfold.backward(grad_input.contiguous(), input_sizes, dim, size, step), None, None, None
        return unfold.backward_cpu(grad_input, input_sizes, dim, size, step), None, None, None


def test():
    dim=0
    steps = list(range(1,10))
    sizes = list(range(1, 10))
    input_sizes = [10,13,21,53,95]
    f = lambda x: [shuffle(a) for a in x]
    f([steps, sizes, input_sizes])

    for step in steps:
        for size in sizes:
            for nelemts in input_sizes:
                a = torch.arange(1, nelemts+1).float()
                a.unfold(dim,size,step) # arguments are legal
                a.requires_grad_()
                
                ret = UnfoldFunction.apply(a, dim, size, step)

                ret.sum().backward()
                
                # Real 
                b = torch.arange(1,nelemts+1).float()
                b.requires_grad_()
                ret = b.unfold(dim, size, step)

                ret.sum().backward()
                
                res = torch.allclose(b.grad, a.grad)

                if not res:
                    print(f"Failed! With step {step} size {size} nelemts {nelemts}" )
                    print("TRUE:", b.grad)
                    print("MINE:", a.grad)
                    print("IN:", a)
                    print("OUT:", ret)
                    return
    print("done test")

def bench_old():
    nelem = 1e5
    dim = 0
    size = 10004
    step = 3
    
    a = torch.arange(nelem).float().cuda()
    a.requires_grad_()
    ret = a.unfold(dim, size, step)

    (ret).sum().backward()

    torch.cuda.synchronize()
    print(torch.cuda.max_memory_allocated() / 1e+9, "gb")

def bench_new():
    nelem = 1e5
    dim = 0
    size = 10004
    step = 3

    a = torch.arange(nelem).float().cuda()
    a.requires_grad_()
    ret = UnfoldFunction.apply(a, dim, size, step)

    (ret).sum().backward()

    torch.cuda.synchronize()
    print(torch.cuda.max_memory_allocated() / 1e+9, "gb")
    

def bench():
    nelem = 1e5
    dim = 0
    size = 304
    step = 13

    def new():
        a = torch.arange(nelem).float().cuda()
        a.requires_grad_()
        ret = UnfoldFunction.apply(a, dim, size, step)

        start = time.time()
        (ret).sum().backward()
        torch.cuda.synchronize()
        print(torch.cuda.max_memory_allocated() / 1e+9, "gb")
        end = time.time()
        print("New took", end-start)

    new()    
    torch.cuda.synchronize()
    print("Memory allocated in between", torch.cuda.memory_allocated())
    def old():

        b = torch.arange(nelem).float().cuda()
        b.requires_grad_()
        ret = b.unfold(dim, size, step)
        
        start = time.time()
        (ret).sum().backward()
        torch.cuda.synchronize()
        print(torch.cuda.max_memory_allocated() / 1e+9, "gb")
        end = time.time()
        print("Old took", end-start)
    old()

if __name__ == '__main__':
    bench()