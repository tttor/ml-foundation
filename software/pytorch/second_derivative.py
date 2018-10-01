#!/usr/bin/env python3
# https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

import torch
from torch.autograd import Variable

import numpy as np
import scipy.optimize as sciopt

def main():
    x_0 = torch.tensor([1.3, 0.7, 0.8, 1.9, 1.2])

    tor_hess = tor_rosen_hess(x_0)
    scipy_hess = sciopt.rosen_hess(x_0.numpy())
    torch.allclose(tor_hess, torch.from_numpy(scipy_hess), rtol=1e-5, atol=1e-8)
    print(tor_hess.sum(dim=0))
    print(tor_hess)
    # test()
    # test2()

def test():
    # https://github.com/pytorch/pytorch/pull/1016#issuecomment-299919437
    x = Variable(torch.ones(1), requires_grad=True)
    y = x.pow(3)

    g = torch.autograd.grad(y, x, create_graph=True)
    print(g) # g = 3

    g2 = torch.autograd.grad(g, x)
    print(g2) # g2 = 6

def test2():
    # https://github.com/pytorch/pytorch/pull/1016#issuecomment-299919437
    # ...a trick to compute the diagonal of the Hessian in a single pass?...
    # There's no way to do that in one go. You can only compute hessian vector products in this way, and
    # there's no vector that will give you the diagonal when you multiply it by a matrix.
    x = Variable(torch.FloatTensor([1,2]), requires_grad=True)
    y = x[0].pow(2) * x[1]

    dx, = torch.autograd.grad(y, x, create_graph=True)
    print(dx)  # (4,1)'

    dx_dx1, = torch.autograd.grad(dx, x, grad_outputs=torch.FloatTensor([1,0]), retain_graph=True)
    dx_dx2, = torch.autograd.grad(dx, x, grad_outputs=torch.FloatTensor([0,1]))

    print(dx_dx1)  # (4,2)'
    print(dx_dx2)  # (2,0)'

def tor_rosen_hess(_x):
    x = Variable(_x.data.clone(), requires_grad=True)
    y = tor_rosen(x)

    dx, = torch.autograd.grad(y, x, create_graph=True)

    dx_xall_sum, = torch.autograd.grad(dx, x, grad_outputs=torch.FloatTensor([1,1,1,1,1]), retain_graph=True)
    print(dx_xall_sum)

    dx_x1, = torch.autograd.grad(dx, x, grad_outputs=torch.FloatTensor([1,0,0,0,0]), retain_graph=True)
    dx_x2, = torch.autograd.grad(dx, x, grad_outputs=torch.FloatTensor([0,1,0,0,0]), retain_graph=True)
    dx_x3, = torch.autograd.grad(dx, x, grad_outputs=torch.FloatTensor([0,0,1,0,0]), retain_graph=True)
    dx_x4, = torch.autograd.grad(dx, x, grad_outputs=torch.FloatTensor([0,0,0,1,0]), retain_graph=True)
    dx_x5, = torch.autograd.grad(dx, x, grad_outputs=torch.FloatTensor([0,0,0,0,1]), retain_graph=False)

    hess = torch.stack((dx_x1, dx_x2, dx_x3, dx_x4, dx_x5))
    return hess

def tor_rosen(x):
    # The Rosenbrock function
    # \param x: a 1D tensor, i.e a vector
    return sum( 100*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0 )

if __name__ == '__main__':
    main()
