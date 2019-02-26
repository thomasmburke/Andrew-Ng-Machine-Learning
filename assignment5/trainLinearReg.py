import numpy as np
from linearRegCostFunction import linearRegCostFunction

def trainLinearReg(X, y, lambdaValue, alpha=None, num_iters=None):
    #initialize theta
    m, n = X.shape
    theta = np.zeros((n+1,1))
    if not num_iters:
        num_iters = 3000
    if not alpha:
        alpha = .001
    J_Hist = []
    for i in range(num_iters):
        J, grad = linearRegCostFunction(X, y, theta, lambdaValue)
        J_Hist.append(J)
        theta = theta - (alpha * grad)
    return theta, J_Hist
