import cvxpy as cp
import numpy as np

n = 100
x = cp.Variable(n)
gamma = cp.Parameter(nonneg=True)
data = cp.Parameter(n)

prob = cp.Problem(cp.Minimize(cp.sum_squares(x - data) + gamma * cp.norm1(x)))

# First solve — compiles and caches the problem structure
gamma.value = 0.1
data.value = np.random.randn(n)
prob.solve()

# Subsequent solves — reuses compiled structure, much faster
gamma.value = 1.0
data.value = np.random.randn(n)
prob.solve()