# LBFGS (belongs to Quasi-Newton)

# Variant
* PBQN (Bollapragada, 2018)
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
* https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS
* https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html

# Misc
* https://en.wikipedia.org/wiki/Limited-memory_BFGS
* http://aria42.com/blog/2014/12/understanding-lbfgs
* https://stats.stackexchange.com/questions/315626/the-reason-of-superiority-of-limited-memory-bfgs-over-adam-solver
* https://scicomp.stackexchange.com/questions/25063/low-rank-updates-in-bfgs
* https://scicomp.stackexchange.com/questions/8470/intuitive-motivation-for-bfgs-update
* https://www.reddit.com/r/MachineLearning/comments/4bys6n/lbfgs_and_neural_nets/
  * concerns: memory, stochastic settings
  * https://github.com/keskarnitish/minSQN
* https://www.quora.com/What-does-it-mean-to-have-a-poorly-conditioned-Hessian-matrix
> Thus, the more ill-conditioned the Hessian is, the more numerically unstable its inverse. Any noise in computing the Hessian such as that introduced by using stochastic versions of descent updates or using minibatches amplifies tremendously when the Hessian is inverted. Methods like L-BFGS get around this by maintaining a low-rank approximation of the (inverse) Hessian which is better suited for ill-conditioned problems as well as saves computation and space required to implement second-order optimization.
