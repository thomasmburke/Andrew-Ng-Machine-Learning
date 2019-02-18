import numpy as np

def predictOneVsAll(all_theta, X):
    """
    Using all_theta, compute the probability of X(i) for each class and predict the label

    return a vector of prediction
    """
    m= X.shape[0]
    X = np.hstack((np.ones((m,1)),X))

    predictions = X @ all_theta.T
    return np.argmax(predictions,axis=1)+1
