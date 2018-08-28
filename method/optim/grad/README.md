# Grad descent
* Adaptive learning rate methods
  * Adagrad
  * AdaDelta
  * RMSProp: identical to Adagrad, but the cache variable is a “leaky”
  * Adam: like RMSProp with momentum.

# Misc
* http://ruder.io/deep-learning-optimization-2017/
  * state-of-the-art results for many tasks in computer vision and NLP ... have still been achieved by **plain old SGD with momentum**
  * adaptive learning rate methods converge to different (and less optimal) minima than SGD with momentum.
  * One factor that partially accounts for Adam's poor generalization ability compared with SGD with momentum on some datasets is weight decay.
  * SGD has been shown to require a learning rate **annealing schedule** to converge to a good minimum
  * ... that decaying the learning rate is equivalent to increasing the batch size, while
    the latter allows for increased parallelism
  * ...  that the number of possible local minima grows exponentially with the number of parameters (Kawaguchi, 2016)
