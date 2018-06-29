# Deep learning via Hessian-free optimization
* James Martens
* icml2010

## problem
*  gradient-descent progresses ex-
tremely slowly on deep nets, seeming to halt altogether be-
fore making significant progress, resulting in poor perfor-
mance on the training set (under-fitting).
* using off-the-shelf implementations of HF
  * that they simply don’t work for neural network
training, or are at least grossly impractical.

## observation
* gradient descent is unsuitable for optimizing objectives
that exhibit pathological curvature.
* 2nd-order optimization
methods, which model the local curvature and correct for
it, have been demonstrated to be quite successful on such
objectives.
*  these objectives exhibit patho-
logical curvature making them nearly impossible for
curvature-blind methods like gradient-descent to success-
fully navigate

## idea: Making HF suitable for machine learning problems
* damping
* matrix-vector product
* handling large dataset
* termination condition for CG
* Sharing information across iterations
* CG iteration backtracking
* Preconditioning CG

## setup
* deep auto-encoder problems considered by Hinton & Salakhutdinov (2006)
*  implemented our approach using the GPU-computing
MATLAB package Jacket.
* dataset:
MNIST and FACES, CURVES

## result
* while bad local optima do exist in deep-
networks (as they do with shallow ones) in practice they do
not seem to pose a significant threat, at least not to strong
optimizers like ours.
* Instead of bad local minima, the diffi-
culty associated with learning deep auto-encoders is better
explained by regions of pathological curvature in the ob-
jective function, which to 1st-order optimization methods
resemble bad local minima.
* learn-
ing in deep models can be achieved effectively and effi-
ciently by a completely general optimizer without any need
for pre-training

## background
* important property of Newton’s method
  * scale invariance
    *  behaves the same for any linear rescaling of the parameters.
  * taking the curvature information into account (in the
form of the Hessian),
    * Newton’s method rescales the gradient so it is a much more sensible direction to follow.
* Hessian-free optimization (HF), aka truncated-Newton, which has
been studied in the optimization community for decades
(e.g. Nocedal & Wright, 1999), but never seriously applied
within machine learning.
* HF optimizes qθ (p) by exploiting two simple ideas
  *  first is that for an N -dimensional vector d, Hd can be
easily computed using finite differences at the cost of a sin-
gle extra gradient evaluation via the identity
  * second is that there is a very effective algorithm for
optimizing quadratic objectives (such as qθ (p)) which re-
quires only matrix-vector products with B: the linear con-
jugate gradient algorithm (CG).

## comment
* (?) where is the algor/pseudocode for all the modif needed for deepnet training in section 4?
* (-) code is not shared
