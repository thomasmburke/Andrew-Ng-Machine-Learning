import numpy as np
from sigmoid import sigmoid, sigmoid_gradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaValue):
    # Reconstruct Thetas from the unrolled neural network params
    Theta1 = nn_params[0:(hidden_layer_size *(input_layer_size + 1))].reshape(hidden_layer_size,(input_layer_size + 1))
    Theta2 = nn_params[(hidden_layer_size *(input_layer_size + 1)):].reshape(num_labels,(hidden_layer_size +1))
    m, n = X.shape
