#!/usr/bin/env python3
import torch

x = torch.tensor(5.0)
w = torch.tensor(3.0, requires_grad=True)
p = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(10.0, requires_grad=True)
alpha = torch.tensor(2.0, requires_grad=True)

# NOT work
# RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.
# w.data.add_(alpha) # In-place version of add()
# w.data = w.data + alpha * p
# w.data.copy_(w.data + alpha * p)
# wlist = [w]
# for i, wi in enumerate(wlist):
#     wi = (wi + alpha * p)
# w = wlist[0]

# Work
# w = (w + alpha * p)

# Work
# wlist = [w]
# wlist2 = []
# for i, wi in enumerate(wlist):
#     wi2 = (wi + alpha * p)
#     wlist2.append(wi2)
# w = wlist2[0]

# Work
# wlist = [w]
# wlist2 = []
# for i, wi in enumerate(wlist):
#     wi2 = (wi + alpha * p)
#     wlist2.append(wi2)
# wlist = wlist2[:]
# w = wlist[0]

wlist = [w]
wlist2 = []
for i, wi in enumerate(wlist):
    wi2 = (wi + alpha * p)
    wlist2.append(wi2)

for i in range(len(wlist)):
    wlist[i] = wlist2[i]
w = wlist[0]

# Compute y, loss
y = x * w + b
loss = 15.0 - y
print(loss)

# loss.backward()
grad_alpha, = torch.autograd.grad(loss, alpha, create_graph=True)
print('grad_alpha', grad_alpha)

# print('alpha.grad=', alpha.grad)
# print('alpha.requires_grad=', alpha.requires_grad)
# print(alpha.grad_fn)

# print('w.grad=', w.grad)
# print(w.grad_fn)

# print('b.grad=', b.grad)
# print(b.grad_fn)

# print('loss.grad=', loss.grad)
# print('loss.requires_grad=', loss.requires_grad)
# print(loss.grad_fn)

# print(y.grad)
# print(y.grad_fn)

