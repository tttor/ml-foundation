#!/usr/bin/env python3
# https://discuss.pytorch.org/t/efficient-computation-of-per-sample-examples/18587/2

import torch
torch.manual_seed(12345)

class LinearWithBatchGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        ctx.save_for_backward(inp, weight, bias)
        return torch.nn.functional.linear(inp, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        inp, weight, bias = ctx.saved_tensors
        grad_input = grad_out @ weight
        grad_weight = (inp.unsqueeze(1) * grad_out.unsqueeze(2))
        grad_bias = grad_out
        return grad_input, grad_weight, grad_bias

def mse_loss(ypred, ytrue):
    # https://discuss.pytorch.org/t/nn-criterions-dont-compute-the-gradient-w-r-t-targets/3693
    return torch.sum((ypred - ytrue).pow(2.0)) / ypred.data.nelement()

# Init #########################################################################
x = torch.randn(5, 1, requires_grad=True)
y = x.pow(2.0) + torch.randn_like(x)
hidden_w = torch.randn(3, 1, requires_grad=True)
hidden_b = torch.randn(3, requires_grad=True)
output_w = torch.randn(1, 3, requires_grad=True)
output_b = torch.randn(1, requires_grad=True)
loss_fn = mse_loss

# Approach: grad is cumulative from all sampless ###############################
ypred = torch.nn.functional.linear(x, hidden_w, hidden_b)
ypred = torch.nn.functional.sigmoid(ypred)
ypred = torch.nn.functional.linear(ypred, output_w, output_b)
loss = loss_fn(ypred, y)

# print(ypred)
# print('ypred.shape=', ypred.shape)
print('loss=', loss)
exit()

gi, ghw, ghb = torch.autograd.grad(loss, [x, hidden_w, hidden_b], retain_graph=True)
# print(ghw)
# print(ghb)

# Approach: grad per sample ####################################################
ypred2 = LinearWithBatchGradFn.apply(x, hidden_w, hidden_b)
ypred2 = torch.nn.functional.sigmoid(ypred2)
ypred2 = LinearWithBatchGradFn.apply(ypred2, output_w, output_b)
loss2 = loss_fn(ypred2, y)

# print(ypred2)
# print('ypred2.shape=', ypred2.shape)
# print('loss2=', loss2)
assert torch.allclose(loss, loss2)
assert torch.allclose(ypred, ypred2)

gi2, ghw2, ghb2 = torch.autograd.grad(loss2, [x, hidden_w, hidden_b])
# print(ghw2)
# print(ghb2)

# Compare ######################################################################
print("grad hidden_w", ghw.shape, ghw2.shape, torch.allclose(ghw2.sum(0), ghw))
print("grad hidden_b", ghb.shape, ghb2.shape, torch.allclose(ghb2.sum(0), ghb))
print("grad inp stays the same for other layers networks", gi.shape, gi2.shape, torch.allclose(gi, gi2))
print('OKAY')
