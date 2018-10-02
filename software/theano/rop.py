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
W = T.dmatrix('W')
x = T.dvector('x')
y = T.dot(x, W)
fy = theano.function([W, x], y)

W_val = [[1, 1], [1, 1]]
x_val = [0, 1]
rfy = fy(W_val, x_val)
print(rfy)

# J, updates = theano.scan(lambda i, y, x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y, x])
# fj = theano.function([W, x], J, updates=updates)
# rfj = fj([W_val, x_val])


# V = T.dmatrix('V'); V_val = [[2, 2], [2, 2]]
# JV = T.Rop(y, W, V)
# f = theano.function([W, x, V], JV)
# r = f(W_val, x_val, V_val)
# print(r)

# v = T.dvector('v')
# Jv = T.Rop(y, W, v)
# f = theano.function([W, v, x], Jv)
# r = f([[1, 1], [1, 1]], [2, 2], [0,1])
# print(r)
