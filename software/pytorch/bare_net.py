#!/usr/bin/env python3
import math

import torch
torch.manual_seed(12345)

def main():
    # Data
    ndata = 10
    input_dim = 1
    x = torch.randn(ndata, input_dim) # random from a standard normal distrib
    y = x.pow(2) + torch.rand(x.size()) # random from a uniform distrib on the interval [0,1)

    # Net
    # net = Net(input_dim, hidden_dim=5, output_dim=1)
    # optim = torch.optim.SGD(net.parameters(), lr=0.2)
    net = BareNet(input_dim, hidden_dim=5, output_dim=1)
    optim = torch.optim.SGD([p for n, p in net.named_parameters() if (('_w' in n) or ('_b' in n))], lr=0.2)

    # print('net.named_parameters()............................................')
    # for n, p in net.named_parameters():
    #     print(n, p.data, p.data.shape, p.requires_grad)

    # train
    loss_fn = torch.nn.MSELoss()
    nepoch = 10

    for epoch in range(nepoch):
        print('### epoch= %d #####################################' % epoch)

        def closure():
            optim.zero_grad()
            ypred = net(x)
            loss = loss_fn(ypred, y)
            loss.backward()

            return loss

        loss = optim.step(closure)
        print('loss= %.7f' % loss)

class BareNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BareNet, self).__init__()

        self.hidden_w = torch.nn.Parameter(torch.zeros(hidden_dim, input_dim))
        self.hidden_b = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.hidden_a = torch.nn.Parameter(torch.zeros(1))
        self.hidden_p = torch.nn.Parameter(torch.zeros(input_dim, hidden_dim), requires_grad=False)
        self.hidden_params = torch.nn.ParameterList([self.hidden_w, self.hidden_b, self.hidden_a, self.hidden_p])

        self.output_w = torch.nn.Parameter(torch.zeros(input_dim, hidden_dim))
        self.output_b = torch.nn.Parameter(torch.zeros(output_dim))
        self.output_a = torch.nn.Parameter(torch.zeros(1))
        self.output_p = torch.nn.Parameter(torch.zeros(hidden_dim, input_dim), requires_grad=False)
        self.output_params = torch.nn.ParameterList([self.output_w, self.output_b, self.output_a, self.hidden_p])

        stdv = 1. / math.sqrt(self.hidden_w.size(1))
        self.hidden_w.data.uniform_(-stdv, stdv)
        self.hidden_b.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.output_w.size(1))
        self.output_w.data.uniform_(-stdv, stdv)
        self.output_b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        hidden_w = self.hidden_w.transpose(0, 1) + (self.hidden_a * self.hidden_p)
        output_w = self.output_w.transpose(0, 1) + (self.output_a * self.output_p)
        y = torch.nn.functional.relu( x.mm(hidden_w) + self.hidden_b )
        y = y.mm(output_w) + self.output_b
        return y

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        y = torch.nn.functional.relu( self.hidden(x) )
        y = self.output(y)
        return y

if __name__ == '__main__':
    main()
