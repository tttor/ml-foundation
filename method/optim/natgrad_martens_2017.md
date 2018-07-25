# New insights and perspectives on the natural gradient method
* James Martens

## abs
* natural grad can be viewed as a type of approximate 2nd-order optimiza-
tion method, where the Fisher information matrix can be viewed as an approximation of the Hessian.

## intro
* for models with very many parameters such as large
  neural networks, computing the natural gradient is impractical due to the extreme size of
  the Fisher information matrix (“the Fisher”).
  * solution: approximate Fisher matrix, eg, KFAC
* Natural gradient descent is generally applicable to the optimization of probabilistic models
  * involves the use of the so-called “natural gradient” in place of the standard gradient,
    * which is defined as the gradient times the inverse of the model’s Fisher information matrix

## 8. The generalized Gauss-Newton matrix

Schraudolph (2002) showed how the idea of the Gauss-Newton matrix can be generalized
to the situation where L(y, z) is any loss function which is convex in z.

## conclusion
The link we have established between natural gradient descent and approximate (stochas-
tic) 2nd-order optimization with the Generalized Gauss-Newton matrix (GGN) provides
intuition for why it might work well with large step-sizes, and gives prescriptions for how
to make it work robustly in practice (by using of damping/regularization techniques).
