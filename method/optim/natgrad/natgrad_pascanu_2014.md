# Revisiting natural gradient for deep networks
* Razvan Pascanu, Yoshua Bengio
* https://arxiv.org/abs/1301.3584
* https://openreview.net/forum?id=vz8AumxkAfz5U
* https://github.com/pascanur/natgrad
* https://www.youtube.com/watch?v=OW-PeoP111E

# intro
* to address the issue of using better optimization techniques for machine learning
  * those which make use of second order information,
    * Hessian- Free Optimization (Martens, 2010) and
      Krylov Subspace Descent (Vinyals and Povey, 2012; Mizutani and Demmel, 2003).
  * those which use the geometry of the underlying parameter manifold
    * e.g. natural gradient descent)
  * those that use the uncertainty in the gradient
    * e.g. TONGA

# 2 Natural gradient descent
* Fisher Information matrix, Equ. 1, 7, 8
* natural gradient descent, Equ. 2
* Given a loss function L parametrized by θ, natural gradient descent attempts to move along the
manifold by correcting the gradient of L according to the local curvature of the KL-divergence
surface
* Algorithm 2 Pseudocode for natural gradient descent algorithm

## 2.1 Adapting natural gradient descent for neural networks
* In order to use natural gradient descent for deterministic neural networks we rely on their probabilistic interpretation
  * For example, the output of an MLP with linear activation function can be interpreted as the mean of
    a conditional Gaussian distribution with a fixed variance, where we condition on the input. Mini-
    mizing the squared error, under this assumption, is equivalent to maximum likelihood.

# 11 Discussion and conclusions
* by employing the
extended Gauss-Newton approximation of the Hessian both Hessian-Free Optimization and Krylov
Subspace Descent can be interpreted as implementing natural gradient descent.
* by adding the previous search direction to the Krylov subspace, KSD does something akin to an ap-
proximation to nonlinear conjugate gradient on the manifold.

# comment
* natural conjugate gradients
* ? confirm this?
>  forward pass (renamed to R-operator in Pearlmutter (1994))

