#!/usr/bin/env python3
# https://discuss.pytorch.org/t/is-there-anyway-to-calculate-gauss-hessian-matrix/10016

import torch
from torch.autograd import Variable
from torch.nn.functional import sigmoid

################################################################################
x = Variable(torch.randn(10,3),requires_grad = False)
w1 = Variable(torch.randn(3,5),requires_grad = True)
w2 = Variable(torch.randn(5,7),requires_grad = True)
z=Variable(torch.randn(1,1),requires_grad = True)
L = (sigmoid(x.mm(w1*z))).mm(w2)
print(L)

zz = Variable(z.data.expand(70, 1, 1), requires_grad=True)
batched_x = x.expand(70, 10, 3)
batched_w1 = w1.expand(70, 3, 5)
batched_w2 = w2.expand(70, 5, 7)
batched_L = (sigmoid(batched_x.bmm(batched_w1*zz))).bmm(batched_w2)
out = batched_L.view(70, 70).trace()
out.backward()
grad = zz.grad.view(10, 7)

print(grad.shape)
