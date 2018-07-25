# Natural Gradient Works Efficiently in Learning
* Shun-ichi Amari
* Neural Computation 10, 251–276 (1998)

## problem
* The stochastic gradient method (Widrow, 1963; Amari, 1967; Tsypkin, 1973;
Rumelhart, Hinton, & Williams, 1986) is a popular learning method in the
general nonlinear optimization framework. The parameter space is not Eu-
clidean but has a Riemannian metric structure in many cases. In these cases,
the ordinary gradient does not give the steepest direction of a target func-
tion; rather, the steepest direction is given by the natural (or contravariant)
gradient.
*  not easy to calculate the natural gradient explicitly in multilayer
perceptrons.
