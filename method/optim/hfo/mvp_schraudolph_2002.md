# Fast Curvature Matrix-Vector Products for Second-Order Gradient Descent
* https://www.mitpressjournals.org/doi/pdf/10.1162/08997660260028683

## abs
* propose a generic method for iteratively approximating various sec-
ond-order gradient steps—-Newton, Gauss-Newton, Levenberg-Mar-
quardt, and natural gradient—-in linear time per iteration, using spe-
cial curvature matrix-vector products that can be computed in O(n)

## intro
* second-order gradient methods do not require C̄−1
explicitly: all they need is its product with the gradient.

* Practical second-order methods therefore prefer measures of curvature that are better
behaved, such as the outer product (Gauss-Newton) approximation of the
Hessian, a model-trust region modification of the same (Levenberg, 1944;
Marquardt, 1963), or the Fisher information.
  * because: Unfortunately, Newton’s method has severe stability problems when
    used in nonlinear systems, stemming from the fact that the Hessian may
    be ill-conditioned and does not guarantee positive definiteness.

* here:
  we define these matrices in a maximum likelihood framework for
  regression and classification and describe O(n) algorithms for computing
  the product of any (approximated) curvature matrix with an arbitrary vector for neural network architectures.

## 2 Definitions and Notation
TODO

## 3 Extended Gauss-Newton Approximation
* Practical second-order gradient methods
  should therefore use approximations or modifications of the Hessian that
  are known to be reasonably well behaved, with positive semidefiniteness
  as a minimum requirement.
  * because: For nonlinear systems,
    H̄ is not necessarily positive definite, so Newton’s method may diverge or
    even take steps in uphill directions.

* the Fisherinformation matrix F̄ (Amari, 1998),
  * which, being a quadratic form, is positive semidefinite by definition.
  * ignores all second-order interactions between system parameters,
    thus throwing away potentially useful curvature information.
