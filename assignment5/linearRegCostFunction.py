import numpy as np

def linearRegCostFunction(X, y, theta, lambdaValue):
    # Calculate Regularized linear regression cost
    m, n = X.shape
    X = np.hstack((np.ones((m,1)),X))
    J = 0
    #print('shape of X: {}'.format(X.shape))
    #print('shape of theta: {}'.format(theta.shape))
    J = sum(np.square((X @ theta) - y))
    J = J * (1/(2*m))
    # remove bias unit in theta
    J_reg = np.sum(np.square(theta[1:]))
    J_reg = J + (lambdaValue/(2*m))*J_reg
    
    # Calculate Regularized Linear Regression Gradient Descent
    grad1 = 1/m * X.T @ ((X @ theta) - y)
    grad2 = 1/m * X.T @ ((X @ theta) - y) + lambdaValue / m * theta
    grad = np.vstack((grad1[0],grad2[1:]))

    return J_reg, grad
