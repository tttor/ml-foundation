#!/usr/bin/env python3
import torch

################################################################################
#https://towardsdatascience.com/getting-started-with-pytorch-part-1-understanding-how-automatic-differentiation-works-5008282073ec

# Define the leaf variables of the graph
a = torch.tensor(4.0)
weights = [torch.tensor(i, requires_grad=True) for i in [2.0, 5.0, 9.0, 7.0]]
w1, w2, w3, w4 = weights

alpha = torch.tensor(0.5, requires_grad=True)
w1.data.add_(alpha)
w2.data.add_(alpha)
w3.data.add_(alpha)
w4.data.add_(alpha)

# Create the graph
b = w1 * a
c = w2 * a
d = w3 * b + w4 * c
L = (10 - d)

# Compute the Gradients
L.backward() # equivalently: L.backward(torch.tensor(1.0))

# for i, w in enumerate(weights, start=1):
#     print(w.grad)
#     print(w.grad_fn) # grad_fn is None because w is a leaf node

print(alpha)
print(alpha.requires_grad)
print(alpha.grad)
exit()

################################################################################
#https://towardsdatascience.com/getting-started-with-pytorch-part-1-understanding-how-automatic-differentiation-works-5008282073ec

# Define the leaf variables of the graph
a = torch.tensor(4.0)
weights = [torch.tensor(i, requires_grad=True) for i in [2.0, 5.0, 9.0, 7.0]]
w1, w2, w3, w4 = weights

# Create the graph
b = w1 * a
c = w2 * a
d = w3 * b + w4 * c
L = (10 - d)

# Compute the Gradients
L.backward() # equivalently: L.backward(torch.tensor(1.0))

for i, w in enumerate(weights, start=1):
    print(w.grad)
    print(w.grad_fn) # grad_fn is None because w is a leaf node

# PyTorch’s default behavior doesn’t allow you to access gradients of non-leaf nodes
# You can override it by calling .retain_grad()
print(b.grad)
print(b.grad_fn)

################################################################################
# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# x = torch.ones(2, 2, requires_grad=True)
# y = x + 2
# z = y * y * 3
# out = z.mean()

# out.backward()
# print(x.grad)
# print(y.grad_fn)
# print(out)

# x = torch.randn(3, requires_grad=True)
# x = torch.tensor([2.0, 5.0], requires_grad=True)
# y = x * 2
# print(x)
# print(y)

# y.backward( torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float) )
# print(x.grad)
