# pytorch
* https://discuss.pytorch.org/t/roadmap-for-torch-and-pytorch/38
* https://github.com/ritchieng/the-incredible-pytorch

## tutor
* https://github.com/pytorch/examples
* https://github.com/yunjey/pytorch-tutorial
* https://github.com/vinhkhuc/PyTorch-Mini-Tutorials
* https://github.com/hunkim/PyTorchZeroToAll
* https://github.com/MorvanZhou/PyTorch-Tutorial
* https://jhui.github.io/2018/02/09/PyTorch-Variables-functionals-and-Autograd/
* https://jhui.github.io/2018/02/09/PyTorch-neural-networks/

## init param
* https://pytorch.org/docs/stable/nn.html#torch-nn-init
* https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
* https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073


### overwrite params
* https://discuss.pytorch.org/t/over-writing-weights-of-a-pre-trained-network-like-alexnet/11912
* https://stackoverflow.com/questions/49446785/how-can-i-update-the-parameters-of-a-neural-network-in-pytorch

### higher order derivatives
* https://discuss.pytorch.org/t/is-there-any-way-to-get-second-order-derivative-or-hessian-matrix/1149
  * https://github.com/pytorch/pytorch/pull/1016#issuecomment-299919437
* https://github.com/pytorch/pytorch/releases/tag/v0.2.0
* https://discuss.pytorch.org/t/getting-hessian-matrix-or-more-higher-derivatives/4711
* https://discuss.pytorch.org/t/calculating-the-hessian-with-0-2/7233
* https://discuss.pytorch.org/t/higher-order-derivatives-implementation-explanation/6329

### hessianMatrix-vector product
* https://discuss.pytorch.org/t/calculating-hessian-vector-product/11240
* https://discuss.pytorch.org/t/issues-computing-hessian-vector-product/2709

### gauss-hessian, jacobian matrix, gradient per individual input vector
* https://discuss.pytorch.org/t/is-there-anyway-to-calculate-gauss-hessian-matrix/10016
* https://discuss.pytorch.org/t/efficient-computation-of-per-sample-examples/18587
* https://discuss.pytorch.org/t/quickly-get-individual-gradients-not-sum-of-gradients-of-all-network-outputs/8405
* https://github.com/pytorch/pytorch/issues/7786
* https://github.com/fKunstner/fast-individual-gradients-with-autodiff
* https://arxiv.org/abs/1510.01799

### misc
* https://discuss.pytorch.org/t/what-is-the-difference-between-tensors-and-variables-in-pytorch/4914/6
  * https://pytorch.org/docs/stable/autograd.html#variable-deprecated
* https://discuss.pytorch.org/t/any-alternatives-to-flat-for-tensor/3106
* https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
  * `(r1 - r2) * torch.rand(a, b) + r2` # is uniformly distributed on [r1, r2].
