import numpy as np

def compute_cost(X, y, theta):
    inner = np.power(((X @ theta) - y), 2)
    return np.sum(inner) / (2 * len(X))
