#!/usr/bin/env python3
# https://discuss.pytorch.org/t/efficient-computation-of-per-sample-examples/18587/2

import torch

class LinearWithBatchGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias=None):
        ctx.save_for_backward(inp, weight, bias)
        return torch.nn.functional.linear(inp, weight, bias)
    @staticmethod
    def backward(ctx, grad_out):
        inp, weight, bias = ctx.saved_tensors
        grad_bias = grad_out if bias is not None else None
        return grad_out @ weight, (inp.unsqueeze(1)*grad_out.unsqueeze(2)), grad_bias

linear_with_batch_grad = LinearWithBatchGradFn.apply


inp = torch.randn(5,2, requires_grad=True)
weight = torch.randn(3,2, requires_grad=True)
bias = torch.randn(3, requires_grad=True)

gradw = torch.randn(5,3)

a = torch.nn.functional.linear(inp, weight, bias)
loss = (a*gradw).sum()

gi, gw, gb = torch.autograd.grad(loss, [inp, weight, bias])

a2 = linear_with_batch_grad(inp, weight, bias)
loss2 = (a2*gradw).sum()
gi2, gw2, gb2 = torch.autograd.grad(loss2, [inp, weight, bias])

print("grad weight", gw.shape, gw2.shape, torch.allclose(gw2.sum(0), gw))
# print("grad bias", gb.shape, gb2.shape, torch.allclose(gb2.sum(0), gb))
# print("grad inp stays the same for other layers networks", gi.shape, gi2.shape, torch.allclose(gi, gi2))

print(gw.data)
print(gw2.data)
print(gw2.sum(0))
