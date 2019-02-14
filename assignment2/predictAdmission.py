import numpy as np

def predict(theta, X):
    predictions = np.dot(X,theta)
    return (predictions > 0)
