# HFO: Hessian-free optimization
* aka "Hessian-free" because 
  * we never construct the Hessian matrix explicitly, 
  * we directly compute the Hessian-matrix vector product, where the vector here is the search direction vector.
  * for the Hessian matrix, $\nabla^2 f(x)$, then the Hessian matrix-vector product is given by
    $\big(\nabla^2 f(x) \big) \cdot v = \nabla_x \big( \nabla_x f(x) \cdot v \big)$
* aka "truncated Newton method" because
  * we truncate the linear CG iteration for some `max_cg_iter`
  * the truncated inner linear CG loop approximately solves the linear equation 
    $\nabla^2 f(x) p = \nabla f(x)$, where $p$ is the step direction vector
    * solving above linear equation is equivalent to solving a minimization of local quadratic approximation of $f(x)$
  
# Gauss-Newton matrix vector product
## paper
* 2011: Learning Recurrent Neural Networks with Hessian-Free Optimization, Martens, J. and Sutskever, I.
  * http://www.icml-2011.org/papers/532_icmlpaper.pdf
  * Sec 3.1
* 2012: Efficient Calculation of the Gauss-Newton Approximation of the Hessian Matrix in Neural Networks, Michael Fairbank
* 2002: Fast Curvature Matrix-Vector Products for Second-Order Gradient Descent, Nicol N. Schraudolph
  * Sec 4 Fast Curvature Matrix-Vector Products
* 1994: Fast Exact Multiplication by the Hessian, Barak A. Pearlmutter 

# Misc
* http://andrew.gibiansky.com/blog/machine-learning/gauss-newton-matrix/
* http://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/
* https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
* https://studywolf.wordpress.com/2016/04/04/deep-learning-for-control-using-augmented-hessian-free-optimization/
  * https://github.com/studywolf/blog/blob/master/train_AHF/train_hf.py
