import numpy as np
from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction

def learningCurve(X, y, Xval, yval, lambdaValue):
    m, n = X.shape
    err_train = []
    err_val = []
    for i in range(1, m+1):
        theta = trainLinearReg(X[0:i,:],y[0:i,:],lambdaValue)[0]
        err_train.append(linearRegCostFunction(X[0:i,:], y[0:i,:], theta, lambdaValue)[0])
        err_val.append(linearRegCostFunction(Xval, yval, theta, lambdaValue)[0])
    return err_train, err_val
