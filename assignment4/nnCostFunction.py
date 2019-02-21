import numpy as np
from sigmoid import sigmoid, sigmoid_gradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaValue):
    # Reconstruct Thetas from the unrolled neural network params
    Theta1 = nn_params[0:(hidden_layer_size *(input_layer_size + 1))].reshape(hidden_layer_size,(input_layer_size + 1))
    Theta2 = nn_params[(hidden_layer_size *(input_layer_size + 1)):].reshape(num_labels,(hidden_layer_size +1))
    m, n = X.shape
    # ******Implement Feed Forward Propogation************
    J = 0
    X = np.hstack(tup=(np.ones(shape=(m,1)),X))
    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack(tup=(np.ones(shape=(len(a1),1)),a1))
    a2 = sigmoid(a1 @ Theta2.T)
    y10 = np.zeros((m,num_labels))
    # np.where will return a matrix [mx1]
    # indexing our matrix results in a vector [m,]
    # in order to brodcast whether our y_actual should have a 1 or 0 the shape must match
    # newaxis adds a dimension
    # y10 represents y_actual in a [5000xK] with a 1 in the location of the correct class
    print('np.where(y==1,1,0) shape: {}'.format(np.where(y==1,1,0).shape))
    print('y10[:,0][:,np.newaxis] shape: {}'.format(y10[:,0][:,np.newaxis].shape))
    for i in range(1, num_labels+1):
        y10[:,i-1][:,np.newaxis] = np.where(y==i,1,0)
    # iterate through each classification type and get the error associated with that classification
    # Then sum up all of the class types error into one final cost
    for i in range(num_labels):
        J = J + sum(-y10[:,i] * np.log(a2[:,i]) - (1 - y10[:,i]) * np.log(1 - a2[:,i]))
    # cost without regularization
    J = 1/m * J
    # Now to compute the cost with regularization
    J_reg = np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:]))
    J_reg = lambdaValue / (2 * m) * J_reg
    J = J + J_reg
    return J
