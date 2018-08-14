# Natural Gradient Works Efficiently in Learning
* Shun-ichi Amari
* Neural Computation 10, 251–276 (1998)

## problem
* The stochastic gradient method (Widrow, 1963; Amari, 1967; Tsypkin, 1973;
Rumelhart, Hinton, & Williams, 1986) is a popular learning method in the
general nonlinear optimization framework. 
  * The parameter space is not Eu-
clidean but has a Riemannian metric structure in many cases. In these cases,
the ordinary gradient does not give the steepest direction of a target func-
tion; rather, the steepest direction is given by the natural (or contravariant)
gradient.
*  not easy to calculate the natural gradient explicitly in multilayer
perceptrons.

## result
* The dynamical behavior of
natural gradient online learning is analyzed and is proved to be Fisher
efficient, implying that it has asymptotically the same performance as the
optimal batch estimation of parameters. This suggests that the plateau
phenomenon, which appears in the backpropagation learning algorithm
of multilayer perceptrons, might disappear or might not be so serious
when the natural gradient is used
* that the performance of natural gradient learn-
ing is remarkably good, and it is sometimes free from being trapped in
plateaus, which give rise to slow convergence of the backpropagation learn-
ing method (Saad & Solla, 1995). This suggests that the Riemannian structure
might eliminate such plateaus or might make them not so serious.

## comment
* key concept:
  Euclidean space, Riemannian metric structure
