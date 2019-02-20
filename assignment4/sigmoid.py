import numpy as np

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def sigmoid_gradient(z):
    return (sigmoid(z) * (1-sigmoid))
