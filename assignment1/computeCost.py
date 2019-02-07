import numpy as np

def compute_cost(X, y, theta):
    inner = np.power(((X @ theta) - y), 2)
    return np.sum(inner) / (2 * len(X))


def slow_compute_cost(X, y, theta):
    inner = np.zeros((len(X),1))
    for i in range(0,len(X)):
        inner[i] = np.power(((theta.T @ X[i,:].reshape(2,1)) - y[i]),2)
    return np.sum(inner) / (2 * len(X))
        
if __name__=='__main__':
    slow_compute_cost(np.array([[1], [2], [3]]),np.array([[1], [2], [3]]),np.array([[1], [2], [3]]))

