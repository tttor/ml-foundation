# optim
More on second order. </br>
See also:
* https://github.com/tttor/math-foundation/tree/master/optim
* https://github.com/tttor/rl-foundation/tree/master/method/policy-based/optim

## for deepnet
* hessian-free (following (Martens, 2010))
  * see [hdf_martens_2010.md](hdf_martens_2010.md)
* SdLBFGS @pytorch
  * https://github.com/harryliew/SdLBFGS
* standard @pytorch
  * https://pytorch.org/docs/stable/optim.html
  * L-BFGS
    * https://discuss.pytorch.org/t/lbfgs-not-functioning-the-way-it-is/16705
    * https://discuss.pytorch.org/t/lbfgs-doesnt-seem-to-work-well/9195/2
* kfac @pytorch
  * (not yet implemented in pytorch/optim, now more like an add-on, augment existing optimizers)
  * https://github.com/yaroslavvb/kfac_pytorch/blob/master/kfac_pytorch.py
    * https://medium.com/@yaroslavvb/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0
    * https://github.com/yaroslavvb/kfac_pytorch/blob/master/deep_autoencoder.ipynb
  * https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/algo/kfac.py

## tutor
* http://andrew.gibiansky.com/blog/machine-learning/gauss-newton-matrix/
* http://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/
* http://andrew.gibiansky.com/blog/machine-learning/conjugate-gradient
* https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
* http://ruder.io/deep-learning-optimization-2017/
* https://studywolf.wordpress.com/2016/04/04/deep-learning-for-control-using-augmented-hessian-free-optimization/
  * https://github.com/studywolf/blog/blob/master/train_AHF/train_hf.py
* https://scicomp.stackexchange.com/questions/14513/minimisation-problem-in-thousands-of-dimensions
