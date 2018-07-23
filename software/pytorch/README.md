# pytorch
* https://discuss.pytorch.org/t/roadmap-for-torch-and-pytorch/38
* https://github.com/ritchieng/the-incredible-pytorch

## book
* https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-pytorch
  * https://github.com/PacktPublishing/Deep-Learning-with-PyTorch

## tutor
* https://github.com/pytorch/examples
* https://github.com/yunjey/pytorch-tutorial
* https://github.com/vinhkhuc/PyTorch-Mini-Tutorials
* https://github.com/hunkim/PyTorchZeroToAll
* https://github.com/MorvanZhou/PyTorch-Tutorial
* https://jhui.github.io/2018/02/09/PyTorch-Variables-functionals-and-Autograd/
* https://jhui.github.io/2018/02/09/PyTorch-neural-networks/

### float vs double
* https://discuss.pytorch.org/t/problems-with-weight-array-of-floattensor-type-in-loss-function/381
> ... recommend using floats instead of doubles. It’s the default tensor type in PyTorch.
On GPUs, float calculations are much faster than double calculations.

### params: add, init
* add
  * https://discuss.pytorch.org/t/adding-new-parameters/13534
* init
  * https://pytorch.org/docs/stable/nn.html#torch-nn-init
  * https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
  * https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
* overwrite
  * https://discuss.pytorch.org/t/over-writing-weights-of-a-pre-trained-network-like-alexnet/11912
  * https://stackoverflow.com/questions/49446785/how-can-i-update-the-parameters-of-a-neural-network-in-pytorch

### higher order derivatives
* https://discuss.pytorch.org/t/is-there-any-way-to-get-second-order-derivative-or-hessian-matrix/1149
  * https://github.com/pytorch/pytorch/pull/1016#issuecomment-299919437
* https://github.com/pytorch/pytorch/releases/tag/v0.2.0
* https://discuss.pytorch.org/t/getting-hessian-matrix-or-more-higher-derivatives/4711
* https://discuss.pytorch.org/t/calculating-the-hessian-with-0-2/7233
* https://discuss.pytorch.org/t/higher-order-derivatives-implementation-explanation/6329

### hessian/newton matrix vector product
* https://discuss.  pytorch.org/t/calculating-hessian-vector-product/11240

### gauss-hessian/gauss-newton matrix vector product
* https://discuss.pytorch.org/t/is-there-anyway-to-calculate-gauss-hessian-matrix/10016

### gradient per individual input vector
* https://discuss.pytorch.org/t/efficient-computation-of-per-sample-examples/18587
* https://discuss.pytorch.org/t/quickly-get-individual-gradients-not-sum-of-gradients-of-all-network-outputs/8405
* https://github.com/pytorch/pytorch/issues/7786
* https://github.com/fKunstner/fast-individual-gradients-with-autodiff
* https://arxiv.org/abs/1510.01799

### conjugate gradient
* https://github.com/pytorch/pytorch/issues/1359
* https://github.com/ikostrikov/pytorch-trpo/blob/master/conjugate_gradients.py
* https://www.upwork.com/job/Implement-conjugate-gradient-optimization-algorithm-neural-network-framework_~01f058eabdf455ece6/
* https://github.com/torch/optim/blob/master/doc/algos.md#optim.cg
  * https://github.com/torch/optim/blob/master/cg.lua
  * http://www.gatsby.ucl.ac.uk/~edward/code/minimize/

### misc
* https://discuss.pytorch.org/t/what-is-the-difference-between-tensors-and-variables-in-pytorch/4914/6
  * https://pytorch.org/docs/stable/autograd.html#variable-deprecated
* https://discuss.pytorch.org/t/any-alternatives-to-flat-for-tensor/3106
* https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
  * `(r1 - r2) * torch.rand(a, b) + r2` # is uniformly distributed on [r1, r2].


