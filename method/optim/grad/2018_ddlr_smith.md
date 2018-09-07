# Don't Decay the Learning Rate, Increase the Batch Size
* Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le
* iclr2018
* https://arxiv.org/abs/1711.00489

## problem
* while SGD finds minima that generalize well (Zhang et al., 2016; Wilson et al., 2017),
each parameter update only takes a small step towards the objective
* when we
increase the batch size the test set accuracy often falls

## result
* can often achieve the benefits of decaying the learning rate by instead increasing the batch size
during training

## comment
* `batch` refers to `minibatch`
