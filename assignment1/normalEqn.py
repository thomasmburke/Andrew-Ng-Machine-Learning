import numpy as np

def normal_equation(X,y):
    return np.linalg.inv((X.T @ X)) @ X.T @ y
