Newton-Type Methods for Non-Convex Optimization Under
Inexact Hessian Information
Peng Xu
Farbod Roosta-Khorasani
Michael W. Mahoney

# background
Classically, the analysis of TR method involves obtaining a minimum descent along
two important directions, namely negative gradient and (approximate) negative curva-
ture.

# + TR, CR
The main advantage of these methods is that they are reliably able to take
advantage of the direction of negative curvature and escape saddle points. More specif-
ically, if the Hessian at a saddle point contains a negative eigenvalue, these methods
can leverage the corresponding direction of negative curvature to obtain decrease in the
objective function values.

# ?
? second-order criticality.

# ref
Raghu Bollapragada, Richard Byrd, and Jorge Nocedal. “Exact and Inexact Sub-
sampled Newton Methods for Optimization”. In: arXiv preprint arXiv:1609.08502
(2016)

Andrew R Conn, Nicholas IM Gould, and Philippe L Toint. Trust region methods.
SIAM, 2000.

Andreas Griewank. “The modification of Newton’s method for unconstrained op-
timization by bounding cubic terms”. In: Technical Report NA/12. Department of
Applied Mathematics and Theoretical Physics, University of Cambridge. (1981).
