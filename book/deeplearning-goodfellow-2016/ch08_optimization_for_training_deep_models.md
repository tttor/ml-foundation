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
TODO
