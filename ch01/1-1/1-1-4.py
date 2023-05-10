import numpy as np

A = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
print(b.shape)
print(A * b)
c = np.array([[10], [20]])
print(c.shape)
print(A * c)