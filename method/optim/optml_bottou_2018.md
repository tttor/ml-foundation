# Optimization Methods for Large-Scale Machine Learning
* https://epubs.siam.org/doi/abs/10.1137/16M1080173

# 1. Introduction.
attempts to provide answers for the following questions:
1. How do optimization problems arise in machine learning applications, and
what makes them challenging?
2. What have been the most successful optimization methods for large-scale
machine learning, and why?
3. What recent advances have been made in the design of algorithms, and what
are open questions in this research area?

# 3. Overview of Optimization Methods.
## 3.1. Formal Optimization Problem Statements.
* f is the composition of the loss function $ell$ and the prediction function h.
* Expected Risk, Empirical Risk

## 3.2. Stochastic vs. Batch Optimization Methods.
* The prototypical stochastic optimization method is the stochastic gradient method (SG)
* while each direction −∇fik (wk ) might not be one of descent from wk
  (in the sense of yielding a negative directional derivative for Rn from w k),
  it is a descent direction in expectation, then the sequence {wk } can be guided toward a minimizer of R
* a batch (deterministic) approach: the steepest descent algorithm
  (also referred to as the gradient, batch gradient, or full gradient method)
  * all samples are considered in an iteration.
* Stochastic and batch approaches offer different trade-offs in minimizing empirical risk, in terms of
  * per-iteration costs and
  * expected per-iteration improvement

## 3.4. Beyond SG: Noise Reduction and Second-Order Methods.
* a minibatch SG: one can employ a minibatch approach in which a small subset
  of samples, call it S k ⊆ {1, . . . , n}, is chosen randomly in each iteration,
* classify two main groups as dynamic sample size and gradient aggregation methods,
  both of which aim to improve the rate of convergence from sublinear to linear
* second-order methods:
  to overcome the adverse effects of high nonlinearity and ill-conditioning of the objective function

# 6 Second-Order Methods
* ways to motivate second-order algorithms
  * first-order methods, such as SG and the full gradient method, are not scale invariant
  * each iteration of the form (6.1) or (6.2) chooses the subsequent iterate by first com-
    puting the minimizer of a second-order Taylor series approximation
    (Newton’s method applies successive local rescalings based on minimizing an
    exact second-order Taylor model of F at each iterate.)
* Deterministic (i.e., batch) methods are known to benefit from the use of second-
  order information; e.g., Newton’s method achieves a quadratic rate of convergence
  if w1 is sufficiently close to a strong minimizer [52].
  * On the other hand, stochastic
    methods like the SG method in section 4 cannot achieve a convergence rate that is
    faster than sublinear, regardless of the choice of B; see [1, 104].

## 6.1. Hessian-Free Inexact Newton Methods.
TODO

# 9. Summary and Perspectives.
* **convergence and complexity theory** for the SG provides
  insight into how these guarantees have translated into practical gains.
* the opportunities offered by parallel and distributed computing
  * SG method may NOT be the best suited method for emerging computer architectures.
* second-order techniques, offer the ability to attain improved convergence rates, overcome the ad-
  verse effects of high nonlinearity and ill-conditioning, and exploit parallelism and
  distributed architectures in new ways.
