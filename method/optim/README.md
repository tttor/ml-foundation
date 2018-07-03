# optim
More on second order. </br>
See also:
* https://github.com/tttor/math-foundation/tree/master/optim

## for deepnet
* hessian-free (following (Martens, 2010))
  * https://github.com/drasmuss/hessianfree # own net
  * https://github.com/MoonL1ght/HessianFreeOptimization # tf
  * https://github.com/doomie/HessianFree # theano
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

## forum
* https://discuss.pytorch.org/t/is-there-anyway-to-calculate-gauss-hessian-matrix/10016
* https://discuss.pytorch.org/t/is-there-any-way-to-get-second-order-derivative-or-hessian-matrix/1149/16
* https://discuss.pytorch.org/t/getting-hessian-matrix-or-more-higher-derivatives/4711/4
* https://discuss.pytorch.org/t/hessian-vector-product-implementation/12923
* https://discuss.pytorch.org/t/calculating-the-hessian-with-0-2/7233/7
* https://discuss.pytorch.org/t/calculating-hessian-vector-product/11240/4
* https://discuss.pytorch.org/t/is-there-any-official-way-to-do-netwons-method-in-pytorch/10199/2
* https://discuss.pytorch.org/t/how-to-calculate-the-2nd-derivative-of-the-diagonal-of-the-hessian-matrix-from-a-function/15093/4
* https://discuss.pytorch.org/t/higher-order-derivatives-implementation-explanation/6329/2

## tutor
* http://andrew.gibiansky.com/blog/machine-learning/gauss-newton-matrix/
* http://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/
* http://andrew.gibiansky.com/blog/machine-learning/conjugate-gradient
* https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
* http://ruder.io/deep-learning-optimization-2017/
* https://studywolf.wordpress.com/2016/04/04/deep-learning-for-control-using-augmented-hessian-free-optimization/
  * https://github.com/studywolf/blog/blob/master/train_AHF/train_hf.py

