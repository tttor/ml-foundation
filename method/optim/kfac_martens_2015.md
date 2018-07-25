# Optimizing Neural Networks with Kronecker-factored Approximate Curvature
* James Martens and Roger Grosse
* icml2015
* http://proceedings.mlr.press/v37/martens15-supp.pdf
* http://proceedings.mlr.press/v37/martens15.pdf
* https://arxiv.org/abs/1503.05671
* http://videolectures.net/icml2015_martens_approximate_curvature/
* http://www.cs.toronto.edu/~jmartens/docs/KFAC3-MATLAB.zip

## observation
* to consider methods which don’t rely on first-order methods like CG as
  their primary engines of optimization
* if we had an efficient and direct way to compute the inverse of
  a high-quality non-diagonal approximation to the curvature matrix-vector
  (i.e. without relying on first-order methods like CG)
  * this could potentially yield an optimization method whose updates would be large and powerful like HF’s,
    while being (almost) as cheap to compute as the stochastic gradient.

## idea: Kronecker-factored Approximate Curvature (K-FAC)
* developing an efficiently invertible approximation to a neural network’s Fisher information matrix,

## result
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
*  The reason that HF sees fewer practical applications than SGD are twofold.
  * its updates are much more expensive to compute,
    * as they involve running linear conjugate gradient (CG) for potentially hundreds of iterations,
      each of which requires a matrix-vector product with the curvature matrix
      (which are as expensive to compute as the stochastic gradient on the current mini-batch).
  * HF’s estimate of the curvature matrix must remain fixed while CG iterates, and
    * thus the method is able to go through much less data than SGD can in a comparable amount of time,
      making it less well suited to stochastic optimizations.
