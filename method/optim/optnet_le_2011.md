# On Optimization Methods for Deep Learning
* Quoc V. Le
* icml2011
* https://cs.stanford.edu/~acoates/papers/LeNgiCoaLahProNg11.pdf

## problem
* SGDs are difficult to tune (eg learning rate) and parallelize.
* A weakness of batch L-BFGS and CG, which require
  the computation of the gradient on the entire dataset
  to make an update, is that they do not scale grace-
  fully with the number of examples.

## setup
* focus on comparing L-BFGS, CG and SGDs.
* MNIST dataset
* for CG and L-BFGS, there are two optimization parameters:
  * minibatch size and
  * number of iterations per minibatch.
*  used LBFGS in minFunc by Mark Schmidt and a CG implementation from Carl Rasmussen.

## result
* (LBFGS, CG) > SGD, where
  * L-BFGS: for low dimensional problems, where the number of parameters are relatively small (e.g., convolutional neural networks).
  * For high dimensional problems, CG often does well
* L-BFGS may NOT  be expected to perform well
  * (e.g., if the Hessian is not well approximated with a low-rank estimate)
* LBFGS, CG:  preference of larger minibatch sizes.

## comment
* note:
  number of iterations per minibatch for LBFGS
* ? CG here means nonlinear-CG?
