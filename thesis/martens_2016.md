# Second-order optimization for neural networks
* James Martens

# Chapter 3 Optimization Difficulties and the role of 2nd-order Optimization
## Sources of difficulty
* Local optimization issues
> The activation functions in neural networks are usually chosen to be smooth functions, which facilitates optimiza-
tion since this implies that the network function f (x, θ) will be smooth, and hence h will be as well (assuming the
loss L is), thus allowing general purpose gradient-based optimization methods to be applied, and their associated
convergence theorems to be valid. However, despite being “smooth” in a strictly mathematical sense, the opti-
mization surface for h may contain many sharp turns, deep and narrow valleys, huge variations in curvature along
different directions which will be reflected in badly conditioned Hessian matrices, and other features for which
mathematical non-smoothness can be seen as a limiting case

*  Global optimization issues
> Owing to the complex way in which θ parameterizes f , the objective h is highly non-convex, and even if we can
efficiently find locally optimal solutions, they will not in general be globally optimal ones. This issue of non-
convexity is a problem that relatively few methods even attempt to deal with, and among the ones that do, none
provide solutions that are completely satisfying. But this shouldn’t be surprising, since we know from classical
neural network theory that finding the global optimum, even for relatively simple feed-forward networks with a
single hidden layer, is an NP-hard problem in general3 (Blum and Rivest, 1988).

* Interaction of global and local optimization
>  it is likely that the quality of the initialization may play
a diminished role for some optimizers (approximate Newton and momentum methods) than for methods such
as standard gradient descent, although there will certainly always be a large difference in performance between
reasonably good initializations and very poor ones

## Approximate Newton/2nd-order approaches
> Standard Newton’s method, where B is given by H and M (δ) is thus a 2nd-order Taylor series approximation
of h, runs into numerous problems when applied to neural network training, owing to the non-convex objective
leading to indefinite quadratic optimization sub-problems whose optima are not well defined, and related issues of
model trust, where unless certain controls are applied, the method will trust its own local quadratic approximations
too much and generate huge/non-nonsensical proposals.

* Why should 2nd-order optimization approaches help?
>  many of the local optimization
issues related to neural network learning can be seen as extreme special cases of problems which arise more
generally in non-linear optimization. For example, tightly coupled parameters with strong local dependencies, and
large variations in scale along different directions in parameter space, are precisely the sorts of issues for which
2nd-order optimization is well suited. Gradient descent on the other hand is well-known to be very sensitive to
such issues, and in order to avoid large oscillations and instability must use a learning rate which is inversely
proportional to the size of the curvature along the highest curvature direction.
> While 2nd-order optimization can potentially increase the effective learning rate along directions which op-
timize long-term behaviors, this won’t necessarily improve optimization performance (locally or globally). For
example, due to the highly nonlinear dependence of the state at a given time-step on a much earlier one, the pro-
posed updates may be very unreliable along such directions, and so following them over the prescribed distance
may do nothing to improve the long-term behavior of the network.

## 3.4 The generalized Gauss-Newton matrix

### 3.4.1 Momentum
The momentum method, as its name suggests, works by storing a “momentum” or “velocity” vector δ and updating θ in the direction of δ instead of the usual gradient

