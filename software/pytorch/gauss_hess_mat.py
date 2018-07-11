#!/usr/bin/env python3
# https://discuss.pytorch.org/t/is-there-anyway-to-calculate-gauss-hessian-matrix/10016

import torch
from torch.autograd import Variable
from torch.nn.functional import sigmoid

################################################################################
z=Variable(torch.randn(1,1),requires_grad = True)
w = Variable(torch.randn(5,1),requires_grad = True)
x = Variable(torch.randn(5,1),requires_grad = False)

zz = Variable(z.data.expand(5, 1), requires_grad=True)

L=(x*w*zz)**2
print(L.sum())

L.sum().backward()

# print(zz.grad)
exit()
################################################################################
x = Variable(torch.randn(3,2),requires_grad = False)
y = Variable(torch.randn(3,1),requires_grad = False)

w1 = Variable(torch.randn(2,5),requires_grad = True)
w2 = Variable(torch.randn(5,1),requires_grad = True)

h_in = x.mm(w1)
h_out = sigmoid(h_in)

y_pred = h_out.mm(w2)

loss_fn = torch.nn.MSELoss(size_average=True, reduce=False)
loss = loss_fn(y_pred, y)
print(loss)


loss.backward()
print(w1.grad.size())

################################################################################
# z=Variable(torch.randn(1,1),requires_grad = True)
# print(w1 * z)
# exit()

# L = (sigmoid(x.mm(w1*z))).mm(w2)

# zz = Variable(z.data.expand(70, 1, 1), requires_grad=True)
# batched_x = x.expand(70, 10, 3)
# batched_w1 = w1.expand(70, 3, 5)
# batched_w2 = w2.expand(70, 5, 7)
# batched_L = (sigmoid(batched_x.bmm(batched_w1*zz))).bmm(batched_w2)
# print(batched_L)
# out = batched_L.view(70, 70).trace()
# out.backward()
# res = zz.grad.view(10, 7)

# print(res.size())
