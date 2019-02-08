import numpy as np

def feature_normalize(X):
    """
    Summary: Feature Scaling - (x-x_avg)/(x_max-x_min)
    Params: X (numpy matrix) - A nxm matrix where n is 2 
        and m is the number of data points
    Return: X_norm (numpy matrix) - a normalized X matrix
    """
    X_avg = np.mean(X, axis=0)
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_norm = (X - X_avg) / (X_max - X_min)
    return X_norm


if __name__=='__main__':
    pass
