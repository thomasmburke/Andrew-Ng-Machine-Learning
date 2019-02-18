import numpy as np
from lrCostFunction import sigmoid

def predict(Theta1, Theta2, X):
    """
    Summary: Create a feed forward propagation algorithm

    Theta1 is a 25x401 matrix
    Theta2 is a 10x26 matrix
    g(x) = sigmoid function
    a^(j) = g(X @ theta^(j-1))
    """

    m, n = X.shape
    X = np.hstack((np.ones((m,1)),X))
    a1 = sigmoid(X @ Theta1.T) # yields a 5000x25 matrix
    # add the bias unit to a2 to ensure dimensions match with the next set of thetas
    a1 = np.hstack((np.ones((m,1)), a1)) # hidden layer
    a2 = sigmoid(a1 @ Theta2.T) # output layer yields a 5000x10 matrix
    return np.argmax(a2,axis=1)+1

