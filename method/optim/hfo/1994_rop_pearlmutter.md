* Fast Exact Multiplication by the Hessian 
* Barak A. Pearlmutter
* https://www.mitpressjournals.org/doi/pdf/10.1162/neco.1994.6.1.147

## problem
*  as the system grows, the diagonal elements of the Hessian become less and less
dominant. Further, the inverse of the diagonal approximation of the Hessian
is known to be a poor approximation to the diagonal of the inverse
Hessian. 

## idea: R{.} technique
* derive an efficient technique for calculating the product of an arbitrary vector v with the Hessian H
* finds this product in O(n) time
and space,' and does not make any approximations
