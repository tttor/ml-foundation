# optim
More on higher order. </br>
See also:
* https://github.com/tttor/math-foundation/tree/master/optim
* https://github.com/tttor/rl-foundation/tree/master/method/policy-based/optim

# for deepnet
* hessian-free opt [(Martens, 2010)](hdf_martens_2010.md)
* SdLBFGS @pytorch
  * https://epubs.siam.org/doi/abs/10.1137/15M1053141
  * https://github.com/harryliew/SdLBFGS
* standard @pytorch
  * https://pytorch.org/docs/stable/optim.html
  * L-BFGS
    * https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS
      * "line search function is not supported yet"
    * https://discuss.pytorch.org/t/lbfgs-not-functioning-the-way-it-is/16705
    * https://discuss.pytorch.org/t/lbfgs-doesnt-seem-to-work-well/9195/2
* kfac @pytorch
  * (not yet implemented in pytorch/optim, now more like an add-on, augment existing optimizers)
  * https://github.com/yaroslavvb/kfac_pytorch/blob/master/kfac_pytorch.py
    * https://medium.com/@yaroslavvb/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0
    * https://github.com/yaroslavvb/kfac_pytorch/blob/master/deep_autoencoder.ipynb
  * https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/algo/kfac.py

# book
* https://mitpress.mit.edu/books/optimization-machine-learning

# tutor
* http://andrew.gibiansky.com/blog/machine-learning/gauss-newton-matrix/
* http://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/
* http://andrew.gibiansky.com/blog/machine-learning/conjugate-gradient
* https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
* http://ruder.io/deep-learning-optimization-2017/
* http://ruder.io/optimizing-gradient-descent/index.html
* https://studywolf.wordpress.com/2016/04/04/deep-learning-for-control-using-augmented-hessian-free-optimization/
  * https://github.com/studywolf/blog/blob/master/train_AHF/train_hf.py
* https://scicomp.stackexchange.com/questions/14513/minimisation-problem-in-thousands-of-dimensions

## natural gradient
* https://wiseodd.github.io/techblog/2018/03/11/fisher-information/
* https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/
* http://kvfrans.com/what-is-the-natural-gradient-and-where-does-it-appear-in-trust-region-policy-optimization/
* http://kvfrans.com/a-intuitive-explanation-of-natural-gradient-descent/

# misc
* https://www.quora.com/Why-second-order-optimization-method-impractical-for-training-neural-network
* https://stats.stackexchange.com/questions/253632/why-is-newtons-method-not-widely-used-in-machine-learning


