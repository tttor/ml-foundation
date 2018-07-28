# 8: Optimization for Training Deep Models

# 8.1 How Learning Differs from Pure Optimization
* notation
  * some performance measure P,
    * is defined with respect to the test set and may also be intractable
  * $J(\theta) = \mathbb{E}_{(x,y) \sim \hat{p}_{data}} L(f(x, \theta), y)$ (8.1)
* in pure opt:
  * minimizing J is a goal in and of itself (directly)
  * is considered to have converged when the gradient becomes very small.
* machine learning opt:
  * optimize P only indirectly.
    * reduce a different cost function J(θ) in the hope that doing so will improve P
  * Training often halts while the surrogate loss function still has large derivatives,
    * the early stopping criterion is based
      on the true underlying loss function, such as 0-1 loss measured on a validation set,
      and is designed to cause the algorithm to halt whenever overfitting begins to occur.
    * training algorithms do not usually halt at a local minimum.
      * Instead, a machine learning algorithm usually minimizes
        a surrogate loss function but halts when a convergence criterion based on early
        stopping (section 7.8) is satisfied.
  * objective function usually decomposes as a sum over the training examples
    * compute each update to the parameters based on an expected value of the cost
      function estimated using only a subset of the terms of the full cost function.

## 8.1.1 Empirical Risk Minimization
* goal of a machine learning algorithm is
  * to reduce the expected generalization error
  * $J^{\star}(\theta) = \mathbb{E}_{(x,y) \sim p_{data}} L(f(x, \theta), y)$ (8.2)
    * aka **risk**
    * the expectation is taken over the **true** underlying distribution
  * in ML we do not the true distrib $p_{data}$
    * otherwise, risk minimization becomes pure optim problem
* trick to go back to pure optim is to minimize the empirical risk
  * to minimize the expected loss on the training set.
  * replacing the true distribution $p(x, y)$ with
    the empirical distribution $\hat{p}(x, y)$ defined by the **training set**.
  * BUT, empirical risk minimization
    * is prone to overfitting
    * is not really feasible.
      * eg if based on gradient,
        many useful loss functions, such as 0-1 loss,
        have no useful derivatives (the derivative is either zero or undefined everywhere).

## 8.1.2 Surrogate Loss Functions and Early Stopping
* in the context of **deep learning**, we **rarely use** empirical risk minimization
  * the quantity that we actually optimize is even more different
    from the quantity that we truly want to optimize,
  * ie via a surrogate loss fn, eg
    * the negative log-likelihood of the correct class
      as a surrogate for the 0-1 loss
      (exactly minimizing expected 0-1 loss is typically intractable
      (exponential in the input dimension))

## 8.1.3 Batch and Minibatch Algorithms
* Most of the properties of the objective function J are expectations over
  the training set.; Equ 8.6
  * compute these expectations by randomly sampling a small number of examples
    from the dataset, then taking the **average over only** those examples.
* motivation for minibatch stochastic gradient descent
  (statistical estimation of the gradient from a small number of samples)
  * Most optimization algorithms
    converge much faster (in terms of total computation, not in terms of number of
    updates) if they are allowed to rapidly compute approximate estimates of the
    gradient rather than slowly computing the exact gradient.
    * recall:
      the standard error of the mean (equation 5.46) estimated from n samples is
      given by $\sigma / \sqrt{n}$
  * may find large numbers of examples that all make very similar contributions
    to the gradient.
* batch or deterministic (gradient) methods
  * is optimization algorithms that use the **entire training set**
    * because they process all the training examples simultaneously in a large batch.
* stochastic method
  * Optimization algorithms that use only a single example at a time
  * in the context of deep learning:
    * somewhere in between:
      using more than one but fewer than all the training examples
    * aka minibatch or minibatch stochastic methods, or simply **stochastic methods**
    * eg: stochastic gradient descent
* Minibatch sizes are generally driven by
  * Small batches can offer a regularizing effect (Wilson and Martinez, 2003),
    * perhaps due to the noise they add to the learning process.
    * Generalization error is often best for a batch size of 1.
  * power of 2 batch sizes to offer better runtime.
    * Some kinds of hardware achieve better runtime with specific sizes of arrays.
  * Larger batches provide a more accurate estimate of the gradient,
    * but with less than linear returns;
      recall the standard error of the mean (equation 5.46): $\sigma / \sqrt{n}$
  * Multicore architectures are usually underutilized by extremely small batches
  * the amount of memory scales with the batch size.
    (If all examples in the batch are to be processed in parallel)
* crucial that the minibatches be selected randomly,
  * in order to
    * make those samples be independent.
    * two subsequent gradient estimates to be independent from each other
  * in practice:
    * to shuffle the order of the dataset once and then store it in shuffled fashion.
  * asynchronous parallel distributed approaches:
    * compute the update that minimizes J(X) for one minibatch of examples X
      at the same time that we compute the update for several other minibatches
* obtain an **unbiased estimator** of the exact gradient of the generalization error
  * by sampling a minibatch of examples $\{x^{(1)}, \ldots, x^{(m)}\}$
    with corresponding targets $y^{(i)}$ from the data-generating distribution pdata,
    * then computing the gradient of the loss with respect to the parameters for that minibatch:
    * ie: $\hat{g} = \frac{1}{m} \nabla_{\theta} \sum_i L\big( f(x^{(i)}, \theta), y^{(i)} \big)$
    * Updating θ in the direction of ĝ performs SGD on the generalization error.
  * this interpretation applies only when examples are **not reused**
    * recall: minibatch stochastic gradient descent follows the gradient of
      the true generalization error (equation 8.2) as long as **no examples are repeated**
* best to make several passes through the training set (aka several epochs),
  unless the training set is extremely large
  * only the first epoch follows the unbiased gradient of the generalization error,
  * BUT the additional epochs usually provide enough benefit due to decreased training error
    (to offset the harm they cause
    by increasing the gap between training error and test error.)
* When using an extremely large training set,
  * to use each training example only once or
    even to make an incomplete pass through the training set
  * then overfitting is not an issue,
    * so underfitting and computational efficiency become the predominant concerns.

# 8.2 Challenges in Neural Network Optimization
## 8.2.1 Ill-Conditioning
* ill-conditioning of the Hessian matrix H
* causing SGD to get “stuck” in the sense that even very small steps increase the cost function
* Figure 8.1 shows an example of
  * the gradient **increasing significantly** during the **successful training** of a neural network.

## 8.2.2 Local Minima
* nearly any deep model is essentially guaranteed to have an extremely large number of local minima.
  * have multiple local minima because of the model identifiability, eg weight space symmetry
  * BUT this is not necessarily a major problem
    * because all these local minima arising from nonidentifiability are equivalent to
      each other in cost function value
  * If local minima with high cost (in comparison to the global minimum) are common,
    this could pose a serious problem for gradient-based optimization algorithms.
* experts now suspect that, for sufficiently large neural networks,
  * most local minima have a low cost function value, and
  * that it is **not important** to find a true global minimum
  * to find a point in parameter space that has **low but not minimal** cost
*  A test that can rule out local minima as the problem is
  * plotting the norm of the gradient over time.
    * If the norm of the gradient does NOT shrink to insignificant size,
      * the problem is neither local minima nor any other kind of critical point.
  * In high-dimensional spaces: very difficult
    *  as Many structures other than local minima also have small gradients

## 8.2.3 Plateaus, Saddle Points and Other Flat Regions
* in low dimensional spaces,
  * local minima are common.
* In higher-dimensional spaces,
  * local minima are rare, and
  * saddle points are more common.
* gradient descent
  * empirically seems able to escape saddle points in many cases.
  *  but the situation may be different for more realistic uses of gradient descent
* For Newton’s method,
  * Without appropriate modification, it can jump to a saddle point.
    * since Newton’s method is designed to solve for a point where the gradient is zero.
  * The proliferation of saddle points in high-dimensional spaces presumably explains
    why second-order methods have not succeeded in replacing gradient descent for neural network training
    * Dauphin et al. (2014) introduced a saddle-free Newton method

## 8.2.4 Cliffs and Exploding Gradients
* cliffs
  * are extremely steep regions
  * result from the multiplication of several large weights together.
* On the face of an extremely steep cliff structure,
  * the gradient update step can move the parameters extremely far,
    usually jumping off the cliff structure altogether
* solution: the gradient clipping heuristic
  * intervenes to reduce the step size,
  * making it less likely to go outside the region where
    the gradient indicates the direction of approximately steepest descent.

## 8.2.5 Long-Term Dependencies
* arises when the computational graph becomes extremely deep.
* Repeated application of the same parameters gives rise to especially pronounced difficulties.

## 8.2.6 Inexact Gradients
* have only a noisy or even biased estimate of these quantities.
  * as using a minibatch of training examples to compute the gradient.
  * When the objective function is intractable, typically its gradient is intractable as well.

## 8.2.7 Poor Correspondence between Local and Global Structure
* in practice, neural networks do not arrive at a critical point of any kind.
* suggests research into choosing good initial points for traditional optimization algorithms to use.
  * all of them might be avoided
    * if there exists a region of space connected reasonably directly to
      a solution by a path that local descent can follow, and
    * if we are able to initialize learning within that well-behaved region

## 8.2.8 Theoretical Limits of Optimization
* finding a solution for a network of a given size is intractable,
  * but in practice we can find a solution easily by using a larger network for which
    many more parameter settings correspond to an acceptable solution.
* in the context of neural network training,
  * not care about finding the exact minimum of a function,
  * but seek only to reduce its value sufficiently to obtain good generalization error.
* Developing more realistic bounds on the performance of optimization algorithms therefore
  remains an important goal for machine learning research.

# 8.3 Basic Algorithms
TODO

# 8.4 Parameter Initialization Strategies
TODO

# 8.5 Algorithms with Adaptive Learning Rates
TODO

# 8.6 Approximate Second-Order Methods
