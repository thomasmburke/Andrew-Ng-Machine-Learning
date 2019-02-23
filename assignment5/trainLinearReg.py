import numpy as np
from linearRegCostFunction import linearRegCostFunction

def trainLinearReg(X, y, lambdaValue):
    #initialize theta
    m, n = X.shape
    theta = np.zeros((n+1,1))
    num_iters = 4000
    alpha = .001
    J_Hist = []
    for i in range(num_iters):
        J, grad = linearRegCostFunction(X, y, theta, lambdaValue)
        J_Hist.append(J)
        theta = theta - (alpha * grad)
    return theta, J_Hist
