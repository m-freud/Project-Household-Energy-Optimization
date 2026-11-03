import cvxpy as cp
import numpy as np

# using bounds directly on variables is more efficient than creating separate constraints for them
x = cp.Variable()

# Bad (with separate constraints) — slow:
constraints = [x >= 0, x <= 1]
prob = cp.Problem(cp.Minimize((x - 0.5)**2), constraints)
prob.solve()

# Good (with bounds on variable) — fast:
x = cp.Variable(bounds=(0, 1))
prob = cp.Problem(cp.Minimize((x - 0.5)**2))
prob.solve()
