# Training Deep and Recurrent Networks with Hessian-Free Optimization
* James Martens and Ilya Sutskever
* G. Montavon et al. (Eds.): NN: Tricks of the Trade, 2nd edn.,
  LNCS 7700, pp. 479–535, 2012.Springer-Verlag Berlin Heidelberg 2012

## Intro
* Hessian-Free optimization (HF)
  * uses local quadratic approximations to generate update proposals
    (Like standard Newton’s method)
  * belongs to the broad class of approximate Newton methods, such as
    Newton-CG, CG-Steihaug, Newton-Lanczos, and Truncated Newton
* has been demonstrated that such an approach can work for
  optimizing non-convex functions, eg deep neural networks
  * if carefully designed and implemented
  * given sensible random initializations.

## Feedforward Neural Networks (FNN)
* Given:
  * an input $x$
  * the parameter $\theta$ that determine weight and biases:
    $(W_1, \ldots, W_{\el - 1}, b_1, \ldots, b_{\el - 1})$
* The FNN computes
  * $y_{i+1} = s_i (W_i y_i + b_i)
  * where:
    * $y_1 = x$
    * $y_i$ is the activation of the net
    * $s_i$ is the nonlinear activation fn, eg sigmoid, tanh
* the objective of interest for learning: the training error
  * is obtained by averaging the losses $f(\theta; (x, t))# over
    a set $S$ of input-output pairs (aka training cases)

## 20.3 Recurrent Neural Networks
TODO

## 20.4 Hessian-Free Optimization Basics
* Setting:
  * unconstrained minimization of
    a twice-differentiable objective function
    $f: \mathbb{R}^n \mapsto \mathbb{R}$
    w.r.t. a vector of real-valued parameters $\theta in \mathbb{R}^n$
* 2nd-order optimizers (such as HF)
  * are derived from the classical Newton’s method (a.k.a. the Newton-Raphson method),
    * an approach based on the idea of iteratively optimizing a sequence of
      local quadratic models/approximations of the objective function
      in order to produce updates to $\theta$
    * iteration $k$ produces a new iterate $\theta_k$ by
      minimizing a local quadratic model $M_{k-1} (\delta)$ of
      the objective $f(\theta{k-1} + \delta)$,
      which is formed using gradient and curvature information local to $\theta{k-1}$
* problem:
  for many good choices of Bk−1 , such as the Hessian at θk−1,
  even computing the entire n × n curvature matrix Bk−1 , let
  alone inverting it/solving the system Bk−1 δk = −f (θk−1 ) (at a cost of O(n3 )),
  will be impractical for all but very small neural networks
* The main idea in Truncated-Newton methods (such as HF)
  * to avoid this costly inversion by partially optimizing the quadratic function $M$
    using the linear conjugate gradient algorithm (CG)
* CG:
  * is a specialized optimizer created specifically for a quadratic objectives
  * has the nice property that
    * requires access to matrix-vectors prod ucts with the curvature matrix B which
      can be computed much more efficiently than the entire matrix in many cases
    * has a fixed-size storage overhead of a few n-dimensional vectors.
    * is a very powerful algorithm, which after i iterations, will find the provably
      optimal solution of any convex quadratic function q(x) over the Krylov subspace
  * the main computational expense of the method:
    * the number of matrix vector products
  * not designed to handle this kind of “stochasticity” due to minibatch and
    its theory depends very much on a stable definition of B for concepts like B-conjugacy to even make sense.
    * solution:
      to fix the minibatch used to define B for the entire run of CG.

* The preconditioned CG algorithm
  * The preconditioning matrix P
    * allows CG to operate within a transformed coordinate system and
    * a good choice of P can substantially accelerate the method
* Termination of CG
* damping methods:
  * are designed to encourage the minimizer of M to be somewhere in Rn where M
    remains a good approximation to f .
  *  allow one to optimize M even when B is indefinite
  * becuase: When f is non-convex (as it is with neural networks), B will sometimes be
    indefinite, and so the minimizer of M may not exist.
*  generalized Gauss-Newton matrix (GGN),
  * is also guaranteed to be positive semi-definite, and
  * tends to work much better than the Hessian in practice as a curvature matrix
    when optimizing non-convex objectives.

## 20.5 Exact Multiplication by the Hessian
TODO

## 20.6 The Generalized Gauss-Newton Matrix
* The indefiniteness of the Hessian
  * is problematic for 2nd-order optimization of non-convex functions because
    an indefinite curvature matrix B may result in a quadratic M which is
    not bounded below and thus does not have a minimizer to use as the update
  * potential solutions
    * imposing a trust-region (sec. 20.8.6) will constrain the optimization,
      or a penalty-based damping method (sec. 20.8.1) will effectively add a positive
      semi-definite (PSD) contribution to B which may render it positive definite (PD).
    * truncated Newton methods is to truncate CG as soon as
      it generates a conjugate direction with negative curvature
      * we have not found to be particularly effective for neural network training.
    * use the generalized Gauss-Newton (GGN) matrix
      * is the best solution to the indefiniteness problem
      * is a provably positive semidefinite curvature matrix that can
        be viewed as an approximation to the Hessian

### 20.6.1 Multiplying by the Gauss-Newton Matrix
* need an efficient algo rithm for computing the $Gv$ products

### 20.6.3 Dealing with Non-convex Losses
* The generalized Gauss-Newton matrix construction will not produce
  a positive definite matrix in the case of non-convex loss
  * because the GGN matrix will usually be PSD only when $L''$ is
* can be addressed in one of several ways
  * could formally treat the nonlinearity as being part of F  and
    redefine the loss L
  * to approximate the loss-Hessian a positive definite matrix, which could be done, say,
    * by adding a scaled multiple of the diagonal to $L''$, or
    * by taking the eigen-decomposition of L'' and
      discarding the eigenvectors that have negative eigenvalues.

## 20.13 Tricks and Recipes
* use of the GGN matrix (instead of the Hessian)
* the CG initialization technique
* a well-designed preconditioner
* For feedforward network learning problems under the default parameterization,
  Tikhonov damping often works well
* the use of the progress-based termination criterion for CG
* dynamic adjustment of damping constants (e.g. λ) according to the LM heuristic

## 20.14 Summary
* to use the generalized Gauss-Newton matrix which is guaranteed to be PSD
* updates must be “damped”
* HF tends to require much larger minibatches than are used in SGD
* the only way to be sure is with careful experimentation.
  * Unfortunately, optimization theory has a long way
    to go before being able to predict the performance of a method like HF applied
    to the highly non-convex objectives functions associated with neural networks.


## Comment
* typo: p483:
  minimizer will exist, and will be given by ...this equ...
* comparing Equ. 20.1 with Nocedals Equ 2.6
* how to map algo in sec 20.5 for HVP in what we have in pytorch?
