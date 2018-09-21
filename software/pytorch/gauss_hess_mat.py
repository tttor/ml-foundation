#!/usr/bin/env python3
# https://discuss.pytorch.org/t/is-there-anyway-to-calculate-gauss-hessian-matrix/10016
import torch
from torch.autograd import Variable
from torch.nn.functional import sigmoid

################################################################################
# L is a transformation that takes a vector z as input.
# If your L can operate on multiple vectors at once,
# then you could do something like the following:
# (here, L squares all elements of the input):
z = torch.randn(3); print('z=', z)
x = Variable(z.expand(3, 3), requires_grad=True); print('x=', x)

out = (x ** 2).trace() # replace x ** 2 with L(x)
print('out=', out)

out.backward()
x.grad  # gives the derivatives matrix
print('x.grad=', x.grad)

# ################################################################################
# z = Variable(torch.randn(1,1),requires_grad = True)
# w = Variable(torch.randn(5,1),requires_grad = True)
# x = Variable(torch.randn(5,1),requires_grad = False)
# L = (xwz)**2

# ################################################################################
# x = Variable(torch.randn(10,3),requires_grad = False)
# w1 = Variable(torch.randn(3,5),requires_grad = True)
# w2 = Variable(torch.randn(5,7),requires_grad = True)
# z=Variable(torch.randn(1,1),requires_grad = True)
# L = (sigmoid(x.mm(w1*z))).mm(w2)
# print(L)

# zz = Variable(z.data.expand(70, 1, 1), requires_grad=True)
# batched_x = x.expand(70, 10, 3)
# batched_w1 = w1.expand(70, 3, 5)
# batched_w2 = w2.expand(70, 5, 7)
# batched_L = (sigmoid(batched_x.bmm(batched_w1*zz))).bmm(batched_w2)
# out = batched_L.view(70, 70).trace()
# out.backward()
# grad = zz.grad.view(10, 7)

# print(grad.shape)
