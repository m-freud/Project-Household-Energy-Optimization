https://www.cvxpy.org/tutorial/performance/index.html

some good practices:

minimize object count
-> use vectors if possible

use available bounds for variables

use parameters for repeateed solves

solve(verbose=True)