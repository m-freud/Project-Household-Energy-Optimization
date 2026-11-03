import cvxpy as cp
import numpy as np

m, n = 500, 200
A = np.random.randn(m, n)
b = np.random.randn(m)
x = cp.Variable(n)

# Slow: creates m separate Constraint objects
constraints = []
for i in range(m):
    constraints.append(A[i, :] @ x == b[i])
prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), constraints)
prob.solve()



# Fast: creates a single Constraint object
constraints = [A @ x == b]
prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), constraints)
prob.solve()