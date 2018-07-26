#!/usr/bin/env python3
import torch

ypred = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(10.0, requires_grad=False)

# torch.nn.MSELoss is not multiplied by 0.5
L = 0.5 * (ypred - y).pow(2.0)
print(L)

dL_dypred, = torch.autograd.grad(L, [ypred], create_graph=True)
print(dL_dypred)
assert torch.allclose(dL_dypred, (ypred - y), rtol=1e-03, atol=1e-03)

d2L_dypred, = torch.autograd.grad(dL_dypred, [ypred])
print(d2L_dypred)
