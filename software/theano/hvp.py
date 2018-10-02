#!/usr/bin/env python3
# http://deeplearning.net/software/theano/tutorial/gradients.html#hessian-times-a-vector
import theano
import theano.tensor as T

x = T.dvector('x')
v = T.dvector('v')
y = T.sum(x ** 2)
gy = T.grad(y, x)
Hv = T.Rop(gy, x, v)
f = theano.function([x, v], Hv)

rf = f([4, 4], [2, 2])
print(rf)
