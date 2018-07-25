# Fast Curvature Matrix-Vector Products for Second-Order Gradient Descent

Unfortunately, Newton’s method has severe stability problems when
used in nonlinear systems, stemming from the fact that the Hessian may
be ill-conditioned and does not guarantee positive definiteness. Practical
second-order methods therefore prefer measures of curvature that are better
behaved, such as the outer product (Gauss-Newton) approximation of the
Hessian, a model-trust region modification of the same (Levenberg, 1944;
Marquardt, 1963), or the Fisher information

For nonlinear systems,
H̄ is not necessarily positive definite, so Newton’s method may diverge or
even take steps in uphill directions. Practical second-order gradient methods
should therefore use approximations or modifications of the Hessian that
are known to be reasonably well behaved, with positive semidefiniteness
as a minimum requirement.


* the Fisherinformation matrix F̄ (Amari, 1998),
  * which, being a quadratic form, is positive semidefinite by definition.
  * ignores all second-order interactions between system parameters,
    thus throwing away potentially useful curvature information.
