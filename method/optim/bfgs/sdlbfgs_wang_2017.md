# Stochastic Quasi-Newton Methods for Nonconvex Stochastic Optimization
* https://epubs.siam.org/doi/abs/10.1137/15M1053141
* Improved SdLBFGS:
  * https://arxiv.org/abs/1805.02338v1
  * https://github.com/harryliew/SdLBFGS #pytorch
  * modif:
    * The Hessian in each step is initialized by the identity matrix.
    * Normalize the direction computed in each step using l2 normalization.

## problem
* stochastic quasi-Newton methods for nonconvex stochastic optimization,

## idea: SdLBFGS
* The damping technique was used to preserve the positive definiteness of Hk,
  without requiring the original prob- lem to be convex.
