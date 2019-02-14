import numpy as np

def feature_scaling(X):
    """
    Summary: Normalizing our test score inputs
    Params: X - np.ndarray that is our test score inputs
    Return: Normalized input array
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = ((X - X_mean) / X_std)
    return X_norm, X_mean, X_std
