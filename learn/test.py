import numpy as np

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print("X:")
print(X)
ones = np.ones((X.shape[0], 1))
print(ones)
X = np.hstack((ones, X))
print(X)
