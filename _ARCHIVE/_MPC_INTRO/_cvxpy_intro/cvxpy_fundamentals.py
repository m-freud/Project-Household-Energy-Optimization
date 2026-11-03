import cvxpy as cp

x = cp.Variable()
y = cp.Variable()
z = cp.Variable()
p1 = cp.Parameter(nonneg=True)
p2 = cp.Parameter(nonneg=True)

p1.value = 1
p2.value = 2

objective = cp.Minimize((p1 * x + p2 * y)**2 + cp.abs(z))

prob = cp.Problem(objective, [p1 * x + p2 * y == 1, z >= 0, x >= 0, y >= 0])

prob.solve()

print(f"Optimal value: {prob.value}")
print(f"Optimal x: {x.value}")
print(f"Optimal y: {y.value}")
print(f"Optimal z: {z.value}")
print("status:", prob.status)


prob2 = cp.Problem(cp.Minimize((p1 * x + p2 * y)**2 + cp.abs(z)), [p1 * x + p2 * y == 1, z >= 0, x >= 0, y >= 0])
prob2.solve()

print(f"Optimal value prob 2: {prob2.value}")
print(f"Optimal x prob 2: {x.value}")
print(f"Optimal y prob 2: {y.value}")
print(f"Optimal z prob 2: {z.value}")
print("status prob 2:", prob2.status)