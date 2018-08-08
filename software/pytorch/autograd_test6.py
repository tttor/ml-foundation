#!/usr/bin/env python3
import torch

def relu(x):
    # ReLU nonlinearity, f(x)= max(0,x)
    x[x < 0.] = 0.
    return x

def d_relu(x):
    # https://stats.stackexchange.com/questions/333394/what-is-the-derivative-of-the-relu-activation-function
    dx = torch.ones_like(x)
    dx[x <= 0.] = 0.
    return dx

x = torch.tensor([[ 1.4271],[-1.8701]])
y = torch.tensor([ 0.8405,  1.4533])
w = torch.tensor([[-0.3248]], requires_grad=True)
ypred = torch.nn.functional.relu(x.mm(w))
ypred = ypred.squeeze()
print('ypred=', ypred)

print('Approach 0...')
# $f = relu(lin(w))$, where $lin(w) = wx$ and $lin'(w) = x$
# $df/dw = f'(w) = relu'(lin(w)) lin'(w)$, which is applying chain-rule for composition of functions
# $f'(w) = relu'(wx) x$
for i in range(y.numel()):
    x_i = torch.index_select(x,dim=0,index=torch.tensor(i))
    dypred_dw_i = d_relu(x_i.mm(w)).mm(x_i)
    print(dypred_dw_i)

print('Approach 1a...')
for i in range(y.numel()):
    dypred_dw_i, = torch.autograd.grad(ypred[i], w, retain_graph=True)
    print('dypred_dw_i=', dypred_dw_i)

print('Approach 1b...')
for i in range(y.numel()):
    mask = torch.zeros(y.numel())
    mask[i] = 1
    dypred_dw_i, = torch.autograd.grad(ypred, w, grad_outputs=mask, retain_graph=True)
    print('dypred_dw_i=', dypred_dw_i)
