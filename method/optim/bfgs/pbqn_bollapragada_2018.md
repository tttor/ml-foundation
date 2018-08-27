# A Progressive Batching L-BFGS Method for Machine Learning
* Raghu Bollapragada 1 Dheevatsa Mudigere 2 Jorge Nocedal 1 Hao-Jun Michael Shi 1 Ping Tak Peter Tang 3
* https://arxiv.org/abs/1802.05374

# problem
* The L-BFGS method (Liu & Nocedal, 1989) has traditionally been
  regarded as a batch method in the machine learning community.
  * This is because quasi-Newton algorithms
    need gradients of high quality in order to construct useful
    quadratic models and perform reliable line searches.
* All of this appears to call for a full
  batch approach, but since small batch sizes give
  rise to faster algorithms with better generalization
  properties, L-BFGS is currently not considered
  an algorithm of choice for large-scale machine
  learning applications.
  * a well-tuned implementation of the
    stochastic gradient (SG) method was far more effective on
    large-scale logistic regression applications than the batch
    L-BFGS method,
    * even when taking into account the advan-
    tages of parallelism offered by the use of large batches.
    *  SG is endowed
    with certain regularization properties that are essential in the
    minimization of such complex nonconvex functions
* how to perform this line search when the objective function is stochastic.
  * stochastic line searches are poorly understood
    and rarely employed in practice because they must make
    decisions based on sample function values

# idea
* postulate that the most efficient algorithms
  for machine learning may not reside entirely in the highly
  stochastic or full batch regimes, but should employ a pro-
  gressive batching approach in which the sample size is ini-
  tially small, and is increased as the iteration progresses.
* a pro- gressive batching approach (aka dynamic sampling)
  * the sample size is initially small, and is increased as the iteration progresses.
* combines three basic components
  * progressive batching,
  * a stochastic line search (adaptive steplength selection)
    * the initial steplength is chosen based on statistical
      information gathered during the course of the iteration.
  * stable quasi-Newton updating

## 2.1. Sample Size Selection
* in the context of first-order methods.
Their inner product test determines a sample size such that
the search direction is a descent direction with high prob-
ability.
* for the search direction: to make an acute angle with the true quasi-Newton search direction

## 2.2. The Line Search
* the first trial steplength in the stochastic backtracking line search
  is computed so that the predicted decrease in the expected
  function value is sufficiently large

## 2.3. Stable Quasi-Newton Updates
* when yk is computed using different samples, the updating process
  may be unstable, and hence it seems natural to use the
  same sample at the beginning and at the end of the iteration
  * However, this requires that the gradient be evaluated twice for every batch

## 2.4. The Complete Algorithm
* skip the quasi-Newton update if the following curvature condition is not satisfied

# setup
* two options for computing the curvature vector
  * the multi-batch (MB) approach (19) with 25% sample overlap, and
  * the full overlap (FO) approach (18)

## binary classification problems
* the logistic loss with regularization
* 8 datasets
* compared our algorithm against two other methods:
  * (i) Stochastic gradient (SG) with a batch size of 1;
  * (ii) SVRG (Johnson & Zhang, 2013) with the inner loop length set to N

## net (prelim)
* against SG and Adam
*  batch normalization and dropout,
which (in their current form) are not conducive to the PBQN
approach due to the need for gradient consistency when
evaluating the curvature pairs in L-BFGS.
* consider three network architectures: (i) a small convolu-
tional neural network on CIFAR-10 (C) (Krizhevsky, 2009),
(ii) an AlexNet-like convolutional network on MNIST
and CIFAR-10 (A1 , A2 , respectively) (LeCun et al., 1998;
Krizhevsky et al., 2012), and (iii) a residual network
(ResNet18) on CIFAR-10 (R) (He et al., 2016)
* Table 1: Best test accuracy performance ... over 5 different runs and initializations.
  * competitive
* the steplength computed via (14) is almost always accepted
by the Armijo condition, and typically lies within (0.1, 1).

# result
* Our numerical experience indicates that formula
  (14) is quite effective at estimating the steplength parameter,
  as it is accepted by the backtracking line search for most
  iterations. As a result, the line search computes very few
  additional function values.
* defining the curvature vector using the MB approach is preferable to using the FB approach
* that has good generalization properties, does not expose any free parameters, and has fast convergence

# background
* operate
  * in the purely stochastic setting (which makes quasi-Newton updating difficult) or
  * in the purely batch regime (which leads to generalization problems)

# comment
* Equ 2: should NOT: the inverse of $H_k$?
* Is not this too small minibatch?
> (SG) with a batch size of 1;
  * ans: recall: Generalization error is often best for a batch size of 1
* yes, but under the hood, there are some constants to set
> the method requires almost no parameter tuning, which is possible due to
the incorporation of second-order information
* how most often?
> show that this steplength procedure is effective on a wide
range of applications, as it leads to well scaled steps and
allows for the BFGS update to be performed most of the
time, even for nonconvex problems.
* Note: ... prelim ...
>  a preliminary investigation into the performance of the PBQN algorithm for training neural networks
> quite difficult due to the existence of local
minimizers, some of which generalize poorly.
