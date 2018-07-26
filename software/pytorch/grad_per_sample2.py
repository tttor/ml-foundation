#!/usr/bin/env python3
import torch
torch.manual_seed(12345)

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        y = torch.nn.functional.sigmoid( self.hidden(x) )
        y = self.output(y)
        return y

ndata = 10
input_dim = 1
x = torch.randn(ndata, input_dim)
y = x.pow(2) + torch.rand(x.size())

net = Net(input_dim, hidden_dim=5, output_dim=1)
loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

# ypred = net(x); print(ypred)
# loss = loss_fn(ypred, y); print(loss)

for i in range(ndata):
    y_i = y[i]
    ypred_i = net(x[i])
    loss_i = loss_fn(ypred_i, y_i)
    print('y_i=', y_i)
    print('ypred_i=', ypred_i)
    print('loss_i=', loss_i)

    # Note: ypred is the input of the loss
    dypred_dw_i, = torch.autograd.grad(ypred_i, [net.hidden.weight], create_graph=False)
    dypred_dw_i = torch.squeeze(dypred_dw_i)
    print('dypred_dw_i=', dypred_dw_i)
    print('dypred_dw_i.shape=', dypred_dw_i.shape)

    # Compute dloss/dypred
    dloss_dypred_i, = torch.autograd.grad(loss_i, [ypred_i], create_graph=True)
    print('dloss_dypred_i=', dloss_dypred_i)

    # Compute d^2(loss)/dypred
    d2loss_dypred_i, = torch.autograd.grad(dloss_dypred_i, [ypred_i])
    print('d2loss_dypred_i=', d2loss_dypred_i)

    v_i = torch.randn_like(dypred_dw_i) # dummy
    gv_i = (d2loss_dypred_i * torch.dot(dypred_dw_i, v_i)) * dypred_dw_i
    print('gv_i=', gv_i)
    print('gv_i.shape=', gv_i.shape)

    exit()

