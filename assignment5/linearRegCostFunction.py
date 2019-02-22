import numpy as np

def linearRegCostFunction(X, y, theta, lambdaValue):
    m, n = X.shape
    X = np.hstack((np.ones((m,1)),X))
    J = 0
    J = sum(np.square((X @ theta) - y))
    J = J * (1/(2*m))
    # remove bias unit in theta
    J_reg = np.sum(np.square(theta[1:]))
    J_reg = J + (lambdaValue/(2*m))*J_reg
    return J_reg

