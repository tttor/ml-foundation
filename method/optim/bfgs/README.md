# LBFGS (belongs to Quasi-Newton)

# Variant
* SLBFGS (Zhao, 2018)
* IQN (Mokhtari, 2018)
* SdLBFGS (Wang, 2017)
* SLBFGS (Moritz, 2016)
* SQN: stochastic quasi-Newton (Byrd, 2016)
* RES: Regularized Stochastic BFGS (Mokhtari, 2014)
* SFO (Sohl-Dickstein, 2014)
* oLBFGS: online LBFGS (Nicol Schraudolph, 2007)

# Software
* https://github.com/chokkan/liblbfgs

# Misc
* https://en.wikipedia.org/wiki/Limited-memory_BFGS
* http://aria42.com/blog/2014/12/understanding-lbfgs
* https://stats.stackexchange.com/questions/315626/the-reason-of-superiority-of-limited-memory-bfgs-over-adam-solver
* https://www.reddit.com/r/MachineLearning/comments/4bys6n/lbfgs_and_neural_nets/
  * concerns: memory, stochastic settings
  * https://github.com/keskarnitish/minSQN
* On optimization methods for deep learning, Quoc V. Le
  * https://cs.stanford.edu/~acoates/papers/LeNgiCoaLahProNg11.pdf
  * Limited memory BFGS (L-BFGS) and  Conjugate gradient (CG) with line search
    can significantly simplify and speed up the
    process of pretraining deep algorithms.
  * used LBFGS in minFunc by Mark Schmidt and a CG implementation from Carl Rasmussen
