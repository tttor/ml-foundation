#!/usr/bin/env python3
# https://discuss.pytorch.org/t/get-gradient-of-loss-wrt-step-length-for-step-length-search/21456

import torch
torch.manual_seed(12345)

N, D_in, H, D_out = 10, 2, 5, 1

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Net: simple ##################################################################
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

p1 = torch.randn(D_in, H, requires_grad=True)
p2 = torch.randn(H, D_out, requires_grad=True)

a = torch.randn(1, requires_grad=True)

w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

w1 = w1 + a * p1
w2 = w2 + a * p2

# RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
# w1 += (a * p1)
# w2 += (a * p2)

# Forward pass: compute predicted y
h = x.mm(w1)
h_relu = torch.nn.functional.relu(h)
y_pred = h_relu.mm(w2)

loss = (y_pred - y).pow(2).mean()
# print(loss.item())

# loss.backward()
# print(w1.grad)

# gv, = torch.autograd.grad(loss, w1, create_graph=True)
# print(gv)

ga, = torch.autograd.grad(loss, a, create_graph=True)
# print(ga)

# Net using nn.module ##########################################################
class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(D_in, H)
        self.output = torch.nn.Linear(H, D_out)

    def forward(self, x):
        y = torch.nn.functional.relu( self.hidden(x) )
        y = self.output(y)
        return y

net = Net(2, 5, 1)
loss_fn = torch.nn.MSELoss()

p1 = torch.transpose(p1, 0, 1)
p2 = torch.transpose(p2, 0, 1)


# Got: RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.
print(net.output.weight)
net.hidden.weight = torch.nn.parameter.Parameter(net.hidden.weight + (a * p1))
net.output.weight = torch.nn.parameter.Parameter(net.output.weight + (a * p2))
print(net.output.weight)

# for name, p in net.named_parameters():
#     # Got: RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
#     # if name == 'hidden.weight':
#     #     p.add_(a * p1)
#     # elif name == 'output.weight':
#     #     p.add_(a * p2)
#     # else:
#     #     pass

#     # Got: RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavio
#     if name == 'hidden.weight':
#         p = p + (a * p1)
#     elif name == 'output.weight':
#         p = p + (a * p2)
#     else:
#         pass

#     # TODO: how to update the weight so that we can compute the gradient of loss wrt step-length a?
#     pass

y_pred = net(x)
loss = loss_fn(y_pred, y)
print(loss.item())

# TODO: get ga
# ga, = torch.autograd.grad(loss, a, create_graph=True)
# print(ga)
