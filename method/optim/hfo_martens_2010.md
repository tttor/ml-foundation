# Deep learning via Hessian-free optimization
* James Martens
* icml2010
* https://github.com/drasmuss/hessianfree # own net
* https://github.com/MoonL1ght/HessianFreeOptimization # tf
* https://github.com/doomie/HessianFree # theano

## problem
* gradient-descent progresses extremely slowly on deep nets,
  seeming to halt altogether before making significant progress,
  resulting in poor performance on the training set (under-fitting).
* using off-the-shelf implementations of HF
  * they simply don’t work for neural network training, or
    are at least grossly impractical.

## observation
* gradient descent (curvature-blind) is unsuitable for optimizing objectives
  that exhibit pathological curvature.
* Hessian-free optimization (HF), aka truncated-Newton, which
  has been studied in the optimization community for decades (e.g. Nocedal & Wright, 1999),
  * but never seriously applied within machine learning.

## idea: Making HF suitable for machine learning problems
### 4.1: damping
* Newton-Lanczos methods: very expensive and thus not cost- effective in practice
* used a simple Levenberg-Marquardt style heuristic for adjusting $\lambda$ directly

### 4.2: matrix-vector product
* the product Hd can be computed using finite- differences,
  * subject to numerical problems and
  * also requires the computationally expensive evaluation of non-linear functions
* Schraudolph (2002) generalized Pearlmutter’s method
  * to compute the product $Gd$ where $G$ is the Gauss-Newton approximation to the Hessian.
* reasons for using G instead of H.
  * Gauss-Newton matrix G is guaranteed to be positive semi-definite, even when un-damped,
    * which avoids the problem of neg- ative curvature,
    * thus guaranteeing that CG will work for any positive value of λ.
  * consistently resulted in much better search directions,
    * even in situations where negative curvature was not present.
    * on all of the learning problems we tested,
  * the associated matrix-vector product algorithm for G
    * uses about half the memory and
    * runs nearly twice as fast.

### 4.3: handling large dataset
* use relatively large mini-batches,
  * the optimal size of which grows as the optimization progresses.
* the gradients and log-likelihoods were computed using the entire dataset.
  * while we used mini-batches to compute the Bd products,

### 4.4: termination condition for CG
* to terminate the iterations once the relative per-iteration progress made in
  minimizing φ(x) fell below some tolerance

### 4.5: Sharing information across iterations
* to use the search direction $p_{n-1}$ found by CG in the previous HF iteration as
  the starting point for CG in the current one.

### 4.6: CG iteration backtracking
* While each successive iteration of CG improves the value of p with respect to
  the 2nd-order model qθ(p), these im- provements are not necessarily reflected in
  the value of f(θ +p).

### 4.7: Preconditioning CG
* we found to be particularly effective was the diagonal matrix

## setup
* deep auto-encoder problems considered by Hinton & Salakhutdinov (2006)
* implemented our approach using the GPU-computing MATLAB package Jacket.
* dataset:
  MNIST and FACES, CURVES

## result
* while bad local optima do exist in deep-
networks (as they do with shallow ones) in practice they do
not seem to pose a significant threat, at least not to strong
optimizers like ours.
* Instead of bad local minima, the diffi-
culty associated with learning deep auto-encoders is better
explained by regions of pathological curvature in the ob-
jective function, which to 1st-order optimization methods
resemble bad local minima.
* learn-
ing in deep models can be achieved effectively and effi-
ciently by a completely general optimizer without any need
for pre-training

## background
* general advatage of 2nd order method, cf SGD
  * converge in fewer training epochs
  * require less tweaking of hyperparams, eg. learning rate
* Newton method
  * central idea:
    $f$ can be locally approximated around each $\theta$, up to 2nd-order, by the quadratic:
    * $f(\theta + p) \approx q_{\theta}(p) \equiv f(\theta) + \nabla f(\theta)^T p + \frac{1}{2} p^T B_p$
    * where $B = H(\theta)$ is the Hessian matrix of $f$ at $\theta$
  * impractical on large models
    * due to the quadratic relationship between the size of the Hessian and the number of params in the model
    * $H$ may be indefinite so the quadratic may not have a minimum
    * more practical:
      * the Hessian is damped or re-conditioned so that $B = H + \lambda I$ for $\lambda \ge 0$
      * quasi-Newton
  * important property of Newton’s method
    * scale invariance
      *  behaves the same for any linear rescaling of the parameters.
    * taking the curvature information into account (in the form of the Hessian),
      * Newton’s method rescales the gradient so it is a much more sensible direction to follow.
* Hessian Free (HF, aka truncated Newton, under the class of quasi-Newton)
  optimizes $q_{\theta}(p)$ by exploiting two simple ideas
  * for an N -dimensional vector d,
    $Hd$ can be easily computed using finite differences at the cost of a sin-
    gle extra gradient evaluation via the identity
  * there is a very effective algorithm foroptimizing quadratic objectives (such as qθ (p))
    * which requires only matrix-vector products with B:
      the linear conjugate gradient algorithm (CG).

## comment
* concept
  * conjugate gradient
    * krylov basis
    * terminal condition
    * proconditioning
  * hessian-free: without explicit calculation or storage of the Hessian.
    * nocedal numopt, ch 7: Large-Scale Unconstrained Optimization
    * Gauss-newton matrix, ch 10
* said:
> Being an optimization algorithm, our approach doesn’t deal specifically with the problem of over-fitting, ...,
  and can be handled by the usual methods of regularization.
* (-) code is not shared
* (?) where is the algor/pseudocode for all the modif needed for deepnet training in section 4?
* (?) how is this reduction specifically?
> Finding a good search direction then reduces to minimizing this quadratic with respect to p.
* (?) pretraining? as in (Hinton, Salakhutdinow, 2006)

