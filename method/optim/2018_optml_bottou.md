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
  * ability to achieve a quadratic rate of convergence in the neighborhood
    of a strong local minimizer
* Deterministic (i.e., batch) methods are known to benefit from the use of second-
  order information; e.g., Newton’s method achieves a quadratic rate of convergence
  if w1 is sufficiently close to a strong minimizer [52].
  * On the other hand, stochastic
    methods like the SG method in section 4 cannot achieve a convergence rate that is
    faster than sublinear, regardless of the choice of B; see [1, 104].

## 6.1. Hessian-Free Inexact Newton Methods.
* solve for Newton direction inexactly through an iterative approach such as
  the conjugate gradient (CG) method
  * can enjoy a superlinear rate of convergence
* CG applied to (6.4b) does not require access to the Hessian itself, only Hessian-vector products,
  (thus called: Hessian-free.)

### 6.1.1. Subsampled Hessian-Free Newton Methods.
* observation in the inexact HFO:
  * the iteration is more tolerant to noise in the Hessian estimate than
    it is to noise in the gradient estimate
* On choosing the subsample size
  * If one chooses the subsample size |Sk H | small enough,
    then the cost of each product involving the Hessian
    approximation can be reduced significantly, thus reducing the cost of each CG iteration.
  * On the other hand, one should choose |Sk H | large enough that the curvature
    information captured through the Hessian-vector products is productive.
  * If done appropriately, Hessian subsampling is robust and effective

> When the Hessians are subsampled (i.e., Sk H ⊂ Sk for all k ∈ N), it has NOT been
shown that the rate of convergence is faster than linear; nevertheless, the reduction
in the number of iterations required to produce a good approximate solution is often
significantly lower than if no Hessian information is used in the algorithm.

### 6.1.2. Dealing with Nonconvexity.
* common to employ
a trust region [37] instead of a line search and to introduce an additional condition in
step 5 of Algorithm 6.1: terminate CG if a candidate solution sk is a direction of neg-
ative curvature,
* the most attractive ways of
doing this in the context of machine learning is to employ a (subsampled) Gauss–
Newton approximation to the Hessian
*  there has been much discussion about the role that
negative curvature and saddle points play in the optimization of DNNs

## 6.2. Stochastic Quasi-Newton Methods.
* construct approximations to the Hessian using only gradient
information, and are applicable for convex and nonconvex problems.
*  natural to ask whether quasi-Newton methods can be
extended to the stochastic setting arising in machine learning
* the distinguishing feature of a quasi-
Newton scheme is that the sequence {Hk} is updated dynamically by the algorithm
rather than through a second-order derivative computation at each iterate.
* enjoyslocal superlinear rate of convergence [51], and this with only first-order information
and without the need for any linear system solves (which are required by Newton’s
method for it to be quadratically convergent).
* issues on BFGS:
  * the update (6.11)
yields dense matrices, even when the exact Hessians are sparse, restricting its use to
small and midsize problems
  * common solution: L-BFGS:
   the matrices {Hk } need not be formed explicitly; instead, each
product of the form Hk ∇F (wk ) can be computed using a formula that only requires
recent elements of the sequence of displacement pairs {(sk , vk )} that have been saved
in storage.

### 6.2.1. Deterministic to Stochastic.
* issues on LBFGS
  * Given that SG also has a sublinear
rate of convergence, what benefit, if any, could come from incorporating Hk into
(6.12)?
  * Can the iter-
ation (6.12) yield fast enough progress as to offset this additional per-iteration cost
  * How could such effects (due to noisy grad estimates)
be avoided in the stochastic regime?

### 6.2.2. Algorithms.
* online L-BFGS
* SQN, performs a sequence of iterations of (6.12) with Hk fixed, then computes a new
displacement pair (sk , vk ) with sk defined as in (6.13) and vk set using one of the
strategies outlined above.
* Experience has shown that some gains in perfor
mance can be achieved, but the full potential of the quasi-Newton schemes discussed
above (and potentially others) is not yet known.

## 6.3. Gauss–Newton Methods.
* The primary advantage of Gauss–
Newton is that it constructs an approximation to the Hessian using only first-order
information, and this approximation is guaranteed to be positive semidefinite, even
when the full Hessian itself may be indefinite. The price to pay for this convenient
representation is that it ignores second-order interactions between elements of the
parameter vector w, which might mean a loss of curvature information that could be
useful for the optimization process
* The primary advantage of Gauss–
Newton is that it constructs an approximation to the Hessian using only first-order
information, and this approximation is guaranteed to be positive semidefinite, even
when the full Hessian itself may be indefinite. The price to pay for this convenient
representation is that it ignores second-order interactions between elements of the
parameter vector w, which might mean a loss of curvature information that could be
useful for the optimization process
* The computational cost of the Gauss–Newton method depends on the dimension-
ality of the prediction function. When the prediction function is scalar-valued, the
Jacobian matrix Jh is a single row whose elements are already being computed as an
intermediate step in the computation of the stochastic gradient ∇f (w; ξ). However,
this is no longer true when the dimensionality is larger than one since then computing
the stochastic gradient vector ∇f (w; ξ) does not usually require the explicit compu-
tation of all rows of the Jacobian matrix.
* probability estimation problems often reduce to using logarithmic losses of the form
f (w; ξ) = − log(h(xξ ; w)). The generalized Gauss–Newton matrix then reduces to
...which does not require explicit computation of the Jacobian Jh,
equ (6.17)

## 6.4. Natural Gradient Method.
* By contrast, the natural
  gradient method [5, 6] aims to be invariant with respect to all differentiable and
  invertible transformations. The essential idea consists of formulating the gradient de-
  scent algorithm in the space of prediction functions rather than specific parameters
* quasi-natural-gradient
* An important tool for the study of Riemannian geometries is the characteriza-
  tion of its geodesics, i.e., the shortest paths connecting two points.
  * In a Riemannian space, on the other hand, the shortest path between
    two points can be curved and does not need to be unique.
* the natural gradient algorithm and Newton’s method
  perform very similarly as optimality is approached.
* the numerical computation of the Fisher information matrix G(wk ) in
  large learning systems is generally very challenging.
  * Moreover, estimating the expec-
    tation (6.21) with, say, a Monte Carlo approach is usually prohibitive due to the cost
    of sampling the current density estimate
  * Several authors [119, 99] suggest using instead a subset of training examples and
    computing a quantity of the form ...p284

## 6.5. Methods that Employ Diagonal Scalings
TODO

# 7. Other Popular Methods.
7.1. Gradient Methods with Momentum.
7.2. Accelerated Gradient Methods.
7.3. Coordinate Descent Methods.

### Information Geometry.
TODO
# 9. Summary and Perspectives.
* **convergence and complexity theory** for the SG provides
  insight into how these guarantees have translated into practical gains.
* the opportunities offered by parallel and distributed computing
  * SG method may NOT be the best suited method for emerging computer architectures.
* second-order techniques, offer the ability to attain improved convergence rates, overcome the ad-
  verse effects of high nonlinearity and ill-conditioning, and exploit parallelism and
  distributed architectures in new ways.
