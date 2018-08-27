# New insights and perspectives on the natural gradient method
* James Martens

## abs
* natural grad can be viewed as a type of approximate 2nd-order optimiza-
tion method, where the Fisher information matrix can be viewed as an approximation of the Hessian.

## 1. Introduction and overview
* for models with very many parameters such as large
  neural networks, computing the natural gradient is impractical due to the extreme size of
  the Fisher information matrix (“the Fisher”).
  * solution: approximate Fisher matrix, eg, KFAC
* Natural gradient descent is generally applicable to the optimization of probabilistic models
  * involves the use of the so-called “natural gradient” in place of the standard gradient,
    * which is defined as the gradient times the inverse of the model’s Fisher information matrix
* the Fisher can be cast as an approximation of the Hessian, Sec 4
* The Fisher, which is used in computing the natural gradient direction, is defined as
the covariance of the gradient of the model’s log likelihood function with respect to cases
sampled from its distribution. 
*  conclusion of this analysis is that with parameter averaging
applied, stochastic gradient descent with a constant step-size/learning-rate achieves the
same asymptotic convergence speed as natural gradient descent (and is thus also “Fisher
efficient”), although 2nd-order methods (such as the latter) can enjoy a more favorable
dependence on the starting point, which means that they can make much more progress
given a limited iteration budget.

## 4. KL divergence objectives
* The natural gradient method of Amari (1998) can be potentially applied to any objective
function which measures the performance of some statistical model. 
  * However, it enjoys richer theoretical properties when applied to objective functions based on the KL diver-
    gence between the model’s distribution and the target distribution, or certain approxima-
    tions/surrogates of these.

## 5. Various definitions of the natural gradient and the Fisher information matrix
TODO

## 8. The generalized Gauss-Newton matrix
Schraudolph (2002) showed how the idea of the Gauss-Newton matrix can be generalized
to the situation where L(y, z) is any loss function which is convex in z.

## conclusion
The link we have established between natural gradient descent and approximate (stochas-
tic) 2nd-order optimization with the Generalized Gauss-Newton matrix (GGN) provides
intuition for why it might work well with large step-sizes, and gives prescriptions for how
to make it work robustly in practice (by using of damping/regularization techniques).
