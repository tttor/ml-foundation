#!/usr/bin/env python3
# https://stackoverflow.com/questions/49429147/replace-diagonal-elements-with-vector-in-pytorch

import torch
torch.manual_seed(12345)

n = 5
G = torch.randn(n, n)
v = torch.randn(n)
l = torch.randn(n)

print(G)
print(G.transpose(0, 1))
print(v)
print(l)

print('===')
r = G.mm(torch.diag(l)).mv(G.transpose(0, 1).mv(v))
print(r)

print('===')
s = torch.zeros(n)
for i in range(n):
    loss_i = l[i]
    g_i = torch.index_select(G, 1, torch.tensor(i)) # NOT: g_i = G[:][i] or g_i = g_i = G[i][:]
    g_i = g_i.squeeze()
    s_i = loss_i * torch.dot(g_i, v) * g_i
    s += s_i
print(s)
assert torch.allclose(r, s)


