from computeCost import compute_cost
import numpy as np

def gradient_descent(X, y, theta, alpha, iterations):
    # Generate a matrix to hold all cost function values
    J_history = np.zeros(shape=(iterations, 1))
    for i in range(iterations):
        # Generate a matrix for each steps cost and ensure it is steadily decreasing
        J_history[i] = compute_cost(X, y, theta)
        # Generate predicted y
        y_hat = np.dot(X, theta)
        # Calculate next theta values
        theta = theta - (alpha / len(X)) * np.dot(X.T, y_hat - y)
    return J_history, theta

