import numpy as np

def polyFeatures(X, p):
    # where p is the degree of the polynomial
    m, n = X.shape
    for i in range(2,p+1):
        X = np.hstack((X, (X[:,0]**i)[:,np.newaxis]))
    return X
