# optim
See also:
* https://github.com/tttor/math-foundation/tree/master/optim
* https://github.com/tttor/rl-foundation/tree/master/method/policy-based/optim

# Newton-type methods
* hessian-free opt [(Martens, 2010)](hdf_martens_2010.md)
* KFAC @pytorch
  * (not yet implemented in pytorch/optim, now more like an add-on, augment existing optimizers)
  * https://github.com/yaroslavvb/kfac_pytorch/blob/master/kfac_pytorch.py
    * https://medium.com/@yaroslavvb/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0
    * https://github.com/yaroslavvb/kfac_pytorch/blob/master/deep_autoencoder.ipynb
  * https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/algo/kfac.py
* Krylov Subspace Descent, Oriol Vinyals et al
  * not require the approximation of the Hessian to be PSD, and our method requires fewer heuristics;
    * however, it requires more memory.

## LBFGS (belongs to Quasi-Newton)
* https://stats.stackexchange.com/questions/315626/the-reason-of-superiority-of-limited-memory-bfgs-over-adam-solver
* https://www.reddit.com/r/MachineLearning/comments/4bys6n/lbfgs_and_neural_nets/
  * concerns: memory, stochastic settings
  * https://github.com/keskarnitish/minSQN
* On optimization methods for deep learning, Quoc V. Le
  * https://cs.stanford.edu/~acoates/papers/LeNgiCoaLahProNg11.pdf
  * Limited memory BFGS (L-BFGS) and
    Conjugate gradient (CG) with line search
    can significantly simplify and speed up the
    process of pretraining deep algorithms.
  * used LBFGS in minFunc by Mark Schmidt and a
    CG implementation from Carl Rasmussen

* SdLBFGS @pytorch
  * https://epubs.siam.org/doi/abs/10.1137/15M1053141
  * https://github.com/harryliew/SdLBFGS
* L-BFGS @pytorch
  * https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS
    * "line search function is not supported yet"
  * https://discuss.pytorch.org/t/lbfgs-not-functioning-the-way-it-is/16705
  * https://discuss.pytorch.org/t/lbfgs-doesnt-seem-to-work-well/9195/2

## natural gradient
* Information Geometry and Its Applications:
  * https://link.springer.com/book/10.1007%2F978-4-431-55978-8
* https://wiseodd.github.io/techblog/2018/03/11/fisher-information/
* https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/
* http://kvfrans.com/what-is-the-natural-gradient-and-where-does-it-appear-in-trust-region-policy-optimization/
* http://kvfrans.com/a-intuitive-explanation-of-natural-gradient-descent/
* https://www.reddit.com/r/MachineLearning/comments/2qpf9x/why_is_the_natural_gradient_not_used_more_in/
* Scaling Up Natural Gradient by Sparsely Factorizing the Inverse Fisher Matrix, Roger B. Grosse, Ruslan Salakhutdinov


# Misc
* https://www.quora.com/Why-second-order-optimization-method-impractical-for-training-neural-network
* https://www.quora.com/Why-are-optimization-techniques-like-natural-gradient-and-second-order-methods-L-BFGS-for-eg-not-much-used-in-deep-learning
* https://stats.stackexchange.com/questions/253632/why-is-newtons-method-not-widely-used-in-machine-learning

## book, course
* https://mitpress.mit.edu/books/optimization-machine-learning
* http://www.cs.cornell.edu/courses/cs6787/2017fa/

## tutor
* http://andrew.gibiansky.com/blog/machine-learning/gauss-newton-matrix/
* http://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/
* http://andrew.gibiansky.com/blog/machine-learning/conjugate-gradient
* https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
* http://ruder.io/deep-learning-optimization-2017/
* http://ruder.io/optimizing-gradient-descent/index.html
* https://studywolf.wordpress.com/2016/04/04/deep-learning-for-control-using-augmented-hessian-free-optimization/
  * https://github.com/studywolf/blog/blob/master/train_AHF/train_hf.py
* https://scicomp.stackexchange.com/questions/14513/minimisation-problem-in-thousands-of-dimensions
