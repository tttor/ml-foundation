#!/usr/bin/env python3
# http://deeplearning.net/software/theano/tutorial/gradients.html#computing-the-jacobian
import theano
import theano.tensor as T

x = T.dvector('x')
y = x ** 2
J, updates = theano.scan(lambda i, y, x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y, x])
f = theano.function([x], J, updates=updates)

r = f([4, 4])
print(r)

