# Practical Gauss-Newton Optimisation for Deep Learning
* icml2017
* https://arxiv.org/abs/1706.03662
* http://proceedings.mlr.press/v70/botev17a

## problem
* First-order methods:
  * hyperparameter tuning of the optimisation parameters is
    often a laborious process, eg  initial learning rate and decay schedule
  * pure stochastic gradient descent often struggles to escape ‘valleys’ in the error surface with 
    largely varying magnitudes of curvature,
* Second-order methods, such as Gauss-Newton, have largely been dismissed
  because of
  * their seemingly prohibitive computational cost and
  * potential instability introduced by using mini-batches.
  * for modern neural networks, explicit calculation and storage of the Hessian matrix is infeasible
* even a block diagonal approximation of Gauss-Newton matrix is computationally infeasible, and 
  additional approximations are required. 

## idea:  Approximate Gauss-Newton Method
* develop a recursive block-diagonal approxima-
tion of the Hessian, where each block corresponds to the
weights in a single feedforward layer.
* Approximating the GN Diagonal Blocks

## result
* competitive against state-of-the-art first-order
  optimisation methods, with sometimes signifi-
  cant improvement in optimisation performance.
* our KFRA approximation performs marginally
  better than KFAC. As we demonstrated, this is possibly due
  to the updates of KFRA being more closely aligned with
  the exact Gauss-Newton updates than those of KFAC.
* confirm that second-order methods can perform admirably against
  even well-tuned state-of-the-art first-order
  approaches, while not requiring any hyperparameter tuning.
* When training on a GPU (as is common in prac-
  tice), we also found that second-order methods can perform
  well, although the improvement over first-order methods
  was more marginal.

## background
* Since the Hessian is not guaranteed to be positive semidefinite,
  * two common alternative curvature measures are
    * the Fisher matrix and
    * the Gauss-Newton matrix.
  * Unfortunately, both are computationally infeasible
    * similar to Martens & Grosse (2015),
      we therefore used a block diagonal approximation,
      followed by a factorised Kronecker approximation
* Besides being intractable for large neural networks, the
  Hessian is not guaranteed to be PSD.
  * A Newton update could therefore lead to an increase in the error.
  * A common PSD approximation to the Hessian is the Gauss-Newton (GN) matrix
* KFAC is less generally appli-
cable since it requires the network to define a probabilistic
model on its output. Furthermore, for non-exponential fam-
ily models, the Gauss-Newton and Fisher approaches are in
general different.
* Martens (2010) and Martens & Sutskever (2011) exploited
the fact that full Gauss-Newton matrix-vector products can
be computed efficiently using a form of automatic differ-
entiation. 

## comment
* ?: transfer functions?
  * ans: equiv to activation fn
