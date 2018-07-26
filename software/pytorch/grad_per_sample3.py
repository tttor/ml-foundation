#!/usr/bin/env python3
import torch
torch.manual_seed(12345)

class LinearWithBatchGradFn(torch.autograd.Function):
    def __init__(self, in_dim, out_dim):



    @staticmethod
    def forward(ctx, inp, weight, bias=None):
        ctx.save_for_backward(inp, weight, bias)
        return torch.nn.functional.linear(inp, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        inp, weight, bias = ctx.saved_tensors
        grad_bias = grad_out if bias is not None else None
        return grad_out @ weight, (inp.unsqueeze(1)*grad_out.unsqueeze(2)), grad_bias

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        y = torch.nn.functional.sigmoid( self.hidden(x) )
        y = self.output(y)
        return y

class Net2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.hidden = LinearWithBatchGradFn(input_dim, hidden_dim)
        self.output = LinearWithBatchGradFn(hidden_dim, output_dim)

    def forward(self, x):
        y = torch.nn.functional.sigmoid( self.hidden(x) )
        y = self.output(y)
        return y

ndata = 10
input_dim = 1
x = torch.randn(ndata, input_dim)
y = x.pow(2) + torch.rand(x.size())

net = Net(input_dim, hidden_dim=5, output_dim=1)
loss_fn = torch.nn.MSELoss(reduce=False)

# ypred = net(x); print(ypred)
# loss = loss_fn(ypred, y); print(loss)

gv_sum = torch.zeros(net.hidden.weight.shape[0])# now: only net.hidden.weight

for i in range(ndata):
    y_i = y[i]
    ypred_i = net(x[i])
    loss_i = loss_fn(ypred_i, y_i)
    # print('y_i=', y_i)
    # print('ypred_i=', ypred_i)
    # print('loss_i=', loss_i)

    # Note: ypred is the input of the loss
    dypred_dw_i, = torch.autograd.grad(ypred_i, [net.hidden.weight], create_graph=False)
    dypred_dw_i = torch.squeeze(dypred_dw_i)
    # print('dypred_dw_i=', dypred_dw_i)
    # print('dypred_dw_i.shape=', dypred_dw_i.shape)

    # Compute dloss/dypred
    dloss_dypred_i, = torch.autograd.grad(loss_i, [ypred_i], create_graph=True)
    # print('dloss_dypred_i=', dloss_dypred_i)

    # Compute d^2(loss)/dypred
    d2loss_dypred_i, = torch.autograd.grad(dloss_dypred_i, [ypred_i])
    # print('d2loss_dypred_i=', d2loss_dypred_i)

    # Compute Gauss-Newton matric vector product
    v_i = torch.randn_like(dypred_dw_i) # dummy
    gv_i = (d2loss_dypred_i * torch.dot(dypred_dw_i, v_i)) * dypred_dw_i
    # print('gv_i=', gv_i)
    # print('gv_i.shape=', gv_i.shape)

    gv_sum += gv_i

print("gv_sum=", gv_sum)
