# A Progressive Batching L-BFGS Method for Machine Learning
* Raghu Bollapragada 1 Dheevatsa Mudigere 2 Jorge Nocedal 1 Hao-Jun Michael Shi 1 Ping Tak Peter Tang 3

## problem
* The L-BFGS method (Liu & Nocedal, 1989) has traditionally been
  regarded as a batch method in the machine learning community.
  * This is because quasi-Newton algorithms
    need gradients of high quality in order to construct useful
    quadratic models and perform reliable line searches.

## idea
* postulate that the most efficient algorithms
for machine learning may not reside entirely in the highly
stochastic or full batch regimes, but should employ a pro-
gressive batching approach in which the sample size is ini-
tially small, and is increased as the iteration progresses.
