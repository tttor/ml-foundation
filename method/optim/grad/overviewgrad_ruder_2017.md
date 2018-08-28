# An overview of gradient descent optimization algorithms
* Sebastian Ruder
* https://arxiv.org/abs/1609.04747
* http://ruder.io/optimizing-gradient-descent/index.html
* https://www.slideshare.net/SebastianRuder/optimization-for-deep-learning

# 2 Gradient descent variants
* three variants of gradient descent, which differ in how much data we use to compute the gradient of the objective function.
  * Batch gradient descent:
    computes the gradient of the cost function w.r.t. to the parameters θ for the entire training dataset:
  * Stochastic gradient descent:
    performs a parameter update for each training example
  * Mini-batch gradient descent:
    performs an update for every mini-batch of n training examples

# 3 Challenges
* Choosing a proper learning rate can be difficult
* Learning rate schedules,
  e.g. annealing,
  i.e. reducing the learning rate according to a pre-defined schedule or when the change in
  objective between epochs falls below a threshold.
* same learning rate applies to all parameter updates.
* avoiding getting trapped in their numerous suboptimal local minima
  * the difficulty arises in fact not from local minima but from saddle points,
    i.e. points where one dimension slopes up and another slopes down.

# 4 Gradient descent optimization algorithms
* Momentum
* NAG: Nesterov accelerated gradient
  * effectively look ahead
    by calculating the gradient not w.r.t. to our current parameters θ but w.r.t. the approximate future
    position of our parameters:
* Adagrad
  * adapts the learning rate to the parameters,
    performing larger updates for infrequent and smaller updates for frequent parameters.
  * problem: diminishing learning rates
* Adadelta
  * is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate
  * restricts the windowaccumulated past gradients to some fixed size w,
    (Instead of accumulating all past squared gradients)
  * not even need to set a default learning rate
* RMSprop
  * divides the learning rate by an exponentially decaying average of squared gradient
* Adam (Adaptive Moment Estimation)
  * keeps an exponentially decaying average of past gradients mt, similar to momentum:
    (In addition to storing an exponentially decaying average of past squared gradients like Adadelta and RMSprop)
  * can be viewed as a combination of RMSprop and momentum:
    * RMSprop contributes the exponentially decaying average of past squared gradients vt , while
    * momentum accounts for the exponentially decaying average of past gradients mt
* AdaMax
* Nadam (Nesterov-accelerated Adaptive Moment Estimation)
  * combines Adam and NAG

# 5 Parallelizing and distributing SGD
* running SGD asynchronously is faster, but
  suboptimal communication between workers can lead to poor convergence
* Variant: Hogwild!, Downpour SGD, Delay-tolerant Algorithms for SGD, Elastic Averaging SGD

# 6 Additional strategies for optimizing SGD
* Shuffling and Curriculum Learning
  * to shuffle the training data after every epoch
  * Curriculum Learning: supplying
    the training examples in a meaningful order may actually lead to improved performance and better convergence.
* Batch normalization
  * reestablishes these normalizations
    (normalize the initial values of our parameters by initializing them with zero mean and unit variance)
    for every mini-batch and changes are back-propagated through the operation as well.
  * acts as a regularizer, reducing (and sometimes even eliminating) the need for Dropout.
* Early stopping
  * should thus always
    monitor error on a validation set during training and stop (with some patience) if your validation error
    does not improve enough
* Gradient noise
  * add noise that follows a Gaussian distribution to each gradient update
  * the added noise gives the model more
    chances to escape and find new local minima, which are more frequent for deeper models.

