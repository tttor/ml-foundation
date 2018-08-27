# LBFGS (belongs to Quasi-Newton)
* On optimization methods for deep learning, Quoc V. Le
  * https://cs.stanford.edu/~acoates/papers/LeNgiCoaLahProNg11.pdf
  * Limited memory BFGS (L-BFGS) and
    Conjugate gradient (CG) with line search
    can significantly simplify and speed up the
    process of pretraining deep algorithms.
  * used LBFGS in minFunc by Mark Schmidt and a
    CG implementation from Carl Rasmussen
* SdLBFGS
  * https://epubs.siam.org/doi/abs/10.1137/15M1053141
  * https://github.com/harryliew/SdLBFGS #pytorch
* https://stats.stackexchange.com/questions/315626/the-reason-of-superiority-of-limited-memory-bfgs-over-adam-solver
* https://www.reddit.com/r/MachineLearning/comments/4bys6n/lbfgs_and_neural_nets/
  * concerns: memory, stochastic settings
  * https://github.com/keskarnitish/minSQN
