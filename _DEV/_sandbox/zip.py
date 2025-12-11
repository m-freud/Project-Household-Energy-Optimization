A = [1, 2, 3]
B = ["A", "B", "C"]
C = ["x", "y", "z", "e"]


print(zip(A, B, C))
for a, b, c in zip(A, B, C):
    print(a, b, c)