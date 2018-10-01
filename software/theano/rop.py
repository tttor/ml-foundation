#!/usr/bin/env python3
import theano
import theano.tensor as T

################################################################################
# W = T.dmatrix('W')
# V = T.dmatrix('V')
# x = T.dvector('x')
# y = T.dot(x, W)
# JV = T.Rop(y, W, V)
# f = theano.function([W, V, x], JV)

# r = f([[1, 1], [1, 1]], [[2, 2], [2, 2]], [0,1])
# print(r)

################################################################################
W = T.dmatrix('W'); W_val = [[1, 1], [1, 1]]
x = T.dvector('x'); x_val = [0, 1]

y = T.dot(x, W)
fy = theano.function([W, x], y)
ry = fy(W_val, x_val)
print(ry)

V = T.dmatrix('V'); V_val = [[2, 2], [2, 2]]
JV = T.Rop(y, W, V)
f = theano.function([W, x, V], JV)
r = f(W_val, x_val, V_val)
print(r)

# v = T.dvector('v')
# Jv = T.Rop(y, W, v)
# f = theano.function([W, v, x], Jv)
# r = f([[1, 1], [1, 1]], [2, 2], [0,1])
# print(r)
