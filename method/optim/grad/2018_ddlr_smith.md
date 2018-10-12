# Don't Decay the Learning Rate, Increase the Batch Size
* Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le
* iclr2018
* https://arxiv.org/abs/1711.00489

## problem
* while SGD finds minima that generalize well (Zhang et al., 2016; Wilson et al., 2017),
  each parameter update only takes a small step towards the objective
* when we increase the batch size the test set accuracy often falls

## idea
* reduce the number of parameter updates by increasing the
learning rate and scaling the batch size
* one should interpret SGDintegrating a stochastic differential equation.
* when the learning rate drops by a factor of α, we instead increase the batch size by α.
  * Decaying the learning rate is simulated annealing
* to demonstrate
that decaying learning rate schedules can be directly converted into increasing batch size schedules,
and vice versa; providing a straightforward pathway towards large batch training

## result
* can often achieve the benefits of decaying the learning rate by
  instead increasing the batch size during training
* one can usually
  obtain the same learning curve on both training and test sets by instead increasing
  the batch size during training. This procedure is successful for stochastic gradi-
  ent descent (SGD), SGD with momentum, Nesterov momentum, and Adam
* shown empirically that increasing the batch size and decay-
ing the learning rate are quantitatively equivalent.

## background
*  Large batches can be parallelized across many machines, reducing training time. Unfortunately, when we
increase the batch size the test set accuracy often falls (Keskar et al., 2016; Goyal et al., 2017).


## comment
* `batch` refers to `minibatch`
