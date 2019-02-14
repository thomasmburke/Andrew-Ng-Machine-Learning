import numpy as np
import matplotlib.pyplot as plt
from sigmoid import sigmoid

def compute_logistic_cost(initial_theta, X, y):
    """
    Summary: computes the cost associated with initial_theta
        also computes the first gradient descent of these values
    Params: numpy array theta, X, and y
    """
    m = len(X)
    z = np.dot(X,initial_theta)
    predictions = sigmoid(z)
    error = (-y * np.log(predictions)) - ((1-y) * np.log(1-predictions))
    cost = (1/m) * sum(error)
    grad = 1/m * np.dot(X.T, (predictions - y))
    return cost[0], grad

def compute_cost(initial_theta, X, y):

    m = len(X)
    z = np.dot(X,initial_theta)
    predictions = sigmoid(z)
    error = (-y * np.log(predictions)) - ((1-y) * np.log(1-predictions))
    cost = (1/m) * sum(error)
    return cost
