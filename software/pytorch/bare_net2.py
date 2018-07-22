#!/usr/bin/env python3
# https://github.com/pytorch/pytorch/issues/4631
# https://discuss.pytorch.org/t/adding-parameters-to-nn-module-whats-the-best-way/5633

import torch
torch.manual_seed(12345)

class Module1(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Module1, self).__init__()

        self.w_i = torch.nn.Parameter(torch.FloatTensor(input_size, hidden_size))


class Module2(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Module2, self).__init__()

        # self.weights = {}
        # self.weights = torch.nn.ModuleDict() # not available in v0.4.0
        # self.weights = torch.nn.ModuleDict()
        # self.weights['w_i'] = torch.nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        self.weight_list = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(input_size, hidden_size))])

m = Module1(5, 5)
print('m parameters:')

for i in m.parameters():
     print(i)

m2 = Module2(5, 5)
print('m2 parameters:')

for i in m2.parameters():
    print(i)
