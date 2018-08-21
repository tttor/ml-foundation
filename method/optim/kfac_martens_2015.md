# Optimizing Neural Networks with Kronecker-factored Approximate Curvature
* James Martens and Roger Grosse
* icml2015
* http://proceedings.mlr.press/v37/martens15.html
* https://arxiv.org/abs/1503.05671
* http://videolectures.net/icml2015_martens_approximate_curvature/
* http://www.cs.toronto.edu/~jmartens/docs/KFAC3-MATLAB.zip
* https://medium.com/@yaroslavvb/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0

## problem
* when using the natural gradient: computing $F^{-1}$  (or its product with $\nabla h$).

## observation
* to consider methods which don’t rely on first-order methods like CG as
  their primary engines of optimization
* if we had an efficient and direct way to compute the inverse of
  a high-quality non-diagonal approximation to the curvature matrix-vector
  (i.e. without relying on first-order methods like CG)
  * this could potentially yield an optimization method whose updates would be large and powerful like HF’s,
    while being (almost) as cheap to compute as the stochastic gradient.

## idea: Kronecker-factored Approximate Curvature (K-FAC)
* is an efficient method for approximating **natural** gradient descent in neural networks
* is based on an efficiently invertible approximation of a neural network’s Fisher information matrix
  * net's Fisher matrix: neither diagonal nor low-rank, and in some cases is completely non-sparse.
* is derived by approximating various large blocks of the Fisher (corresponding to entire layers)
  as being the **Kronecker product** of two much smaller matrices

* developing an efficiently invertible approximation to a neural network’s Fisher information matrix,

## setup
* applied it to the 3 deep-autoencoder optimization prob-
  lems from Hinton and Salakhutdinov (2006), which use
  the “MNIST”, “CURVES”, and “FACES” datasets respectively
* baseline we used the version of SGD with momentum based on Nesterov’s Accelerated Gradient
  * not compare to methods based on diagonal approximations of the curvature matrix
* used a exponentially decayed iterate averaging approach based on Polyak averaging
  * To help mitigate the detrimental effect that the noise in the
stochastic gradient has on the convergence of the baseline
(and to a lesser extent K-FAC)
* report the error on
the training set as opposed to the test set, as we are chiefly
interested in optimization speed and not the generalization
capabilities of the networks themselves.


## result
* only several times more expensive to computethan the plain stochastic gradient, the updates
  produced by K-FAC make much more progress optimizing the objective,
  * which results in an algorithm that can be much faster than stochastic gradient descent with momentum in practice
* works very well in highly stochastic optimization regimes.
  * because the cost of storing and inverting K-FAC’s approximation to the curvature matrix
    does NOT depend on the amount of data used to estimate it
* without the momentum technique developed in Ap-
  pendix F, K-FAC wasn’t significantly faster than the base-
  line (which itself used a strong form of momentum)
*  K-FAC may be much better
suited than the SGD baseline for a massively distributed
implementation, since it would require far fewer synchro-
nization steps (by virtue of the fact that it performs far
fewer iterations)
* much faster in practice than even highly tuned implementations of SGD with momentum
  on certain standard neural network optimization benchmarks
* main advantages of K-FAC over HF are twofold.
  * K-FAC uses an efficiently computable direct solution for the inverse of the curvature matrix and
    * thus avoids the costly matrix-vector products associated with running CG within HF.
  * it can estimate the curvature matrix from a lot of data by using an online exponentially-decayed average,
    * as opposed to relatively small-sized fixed mini-batches used by HF.
    * The cost of doing this is of course the use of an inexact approximation to the curvature matrix.
* works very well in highly stochastic optimization regimes.
  * unlike: Hessian-free optimization which use high-quality non-diagonal curvature matrices
* K-FAC utilizes the special structure of neural networks (unlike SGD or HF),
  * thus: not directly applicable to other neural architectures like RNNs or
    convolutional neural networks (CNNs).

## background
* The reason that HF sees fewer practical applications than SGD are twofold.
  * its updates are much more expensive to compute,
    * as they involve running linear conjugate gradient (CG) for potentially hundreds of iterations,
      each of which requires a matrix-vector product with the curvature matrix
      (which are as expensive to compute as the stochastic gradient on the current mini-batch).
  * HF’s estimate of the curvature matrix must remain fixed while CG iterates, and
    * thus the method is able to go through much less data than SGD can in a comparable amount of time,
      making it less well suited to stochastic optimizations.
* the natural gradient defines the direction in parameter space which gives
  the largest change in the objective per unit of **change in the model**, as measured by the KL-divergence.
  * cf: standard gradient,
    * which can be defined as the direction in parameter space which gives the
      largest change in the objective per unit of **change in the parameters**, as measured by the standard Euclidean metric.
* the Fisher is equivalent to the Generalized Gauss-Newton matrix (GGN) in certain important cases,
  * GGN is a well-known positive semi-definite approximation to the Hessian of the objective function
  * natural gradientbased optimization methods can conversely be viewed as 2nd-order optimization methods
* The expectation of a Kronecker product is, in general, not equal to the Kronecker product of expectations

## comment
* math:
  * https://en.wikipedia.org/wiki/Kronecker_product

* ? why saying:
> CG ... because it is a first-order method.
