# HFO: Hessian-free optimization
* aka "Hessian-free" because 
  * the Hessian matrix is never constructed explicitly, 
  * we directly compute the Hessian-matrix vector product, where the vector here is the search direction vector.
  * for the Hessian matrix, $\nabla^2 f(x)$, then the Hessian matrix-vector product is given by
    $\big(\nabla^2 f(x) \big) \cdot v = \nabla_x \big( \nabla_x f(x) \cdot v \big)$
    (alternatively, this $Hv$ can be computed using R-op, see [this Theano doc](http://deeplearning.net/software/theano/tutorial/gradients.html#hessian-times-a-vector))
* aka "truncated Newton method" because
  * we truncate the linear CG iteration for some `max_cg_iter`
  * the truncated inner linear CG loop 
    * approximately solves the linear equation 
      $\nabla^2 f(x) p = \nabla f(x)$, where $p$ is the step direction vector
    * solving above linear equation is equivalent to solving a minimization of local quadratic approximation of $f(x)$
  
# Generalized-Gauss-Newton matrix-vector product
## paper
* 2011: Learning Recurrent Neural Networks with Hessian-Free Optimization, Martens, J. and Sutskever, I.
  * http://www.icml-2011.org/papers/532_icmlpaper.pdf
  * Sec 3.1
* 2012: Efficient Calculation of the Gauss-Newton Approximation of the Hessian Matrix in Neural Networks, Michael Fairbank
* 2002: Fast Curvature Matrix-Vector Products for Second-Order Gradient Descent, Nicol N. Schraudolph
  * Sec 4 Fast Curvature Matrix-Vector Products
* 1994: Fast Exact Multiplication by the Hessian, Barak A. Pearlmutter 

## pytorch
* https://github.com/pytorch/pytorch/issues/8304
* https://discuss.pytorch.org/t/r-operator-in-pytorch/19335
* https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968
* https://discuss.pytorch.org/t/more-efficient-implementation-of-jacobian-matrix-computation/6960
* https://discuss.pytorch.org/t/calculating-jacobian-in-a-differentiable-way/13275

## misc
* http://deeplearning.net/software/theano/tutorial/gradients.html#r-operator
  * how does the R-op compute the Jacobian under the hood?
> Work is in progress on the optimizations required to compute efficiently the full Jacobian and the Hessian matrix as well as the Jacobian times vector.

```
Signature: T.Rop(f, wrt, eval_points)
Docstring:
Computes the R operation on `f` wrt to `wrt` evaluated at points given
in `eval_points`. Mathematically this stands for the jacobian of `f` wrt
to `wrt` right muliplied by the eval points.  
```
* https://j-towns.github.io/2017/06/12/A-new-trick.html
  * implementing Rop in Theano may be unnecessary.
  * computing generalised Gauss Newton matrix-vector products, upon a new trick: 
    * a method for calculating jvps by composing two reverse mode vjps!
  * note: 
    * R-op uses forward mode AD
    * L-op uses backward mode AD
  * alternative R-op (twice L-op)
```py
def alternative_Rop(f, x, u):
    v = f.type('v')       # Dummy variable v of same type as f
    g = T.Lop(f, x, v)    # Jacobian of f left multiplied by v
    return T.Lop(g, v, u)
```    

# Misc
* http://andrew.gibiansky.com/blog/machine-learning/gauss-newton-matrix/
* http://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/
* https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
* https://studywolf.wordpress.com/2016/04/04/deep-learning-for-control-using-augmented-hessian-free-optimization/
  * https://github.com/studywolf/blog/blob/master/train_AHF/train_hf.py
