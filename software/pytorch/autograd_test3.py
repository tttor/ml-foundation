#!/usr/bin/env python3
import torch

def main():
    def rosenbrock(x):
        return sum( 100*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0 )

    def net(x):
        y = x * w + b
        loss = y.norm()
        return loss

    def phi(alpha):
        return f(x + alpha * p) # equ 3.55, p56

    def grad(fn, x):
        out = fn(x)
        grad_val, = torch.autograd.grad(out, x, create_graph=True)
        return grad_val

    x = torch.tensor([1.3, 0.7, 0.8, 1.9, 1.2], requires_grad=True)
    print('x=', x)

    p = torch.tensor([1.3, 0.7, 0.8, 1.9, 1.2])
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(10.0, requires_grad=True)

    # f = rosenbrock
    f = net

    fx = f(x)
    print('fx=', fx)

    dfx = grad(f, x)
    print('dfx=', dfx)

    alpha = torch.tensor(0.5, requires_grad=True)
    print('alpha=', alpha)

    phi_alpha = phi(alpha)
    print('phi_alpha=', phi_alpha)

    dphi_alpha = grad(phi, alpha)
    print('dphi_alpha=', dphi_alpha)

if __name__ == '__main__':
    main()
