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
    return y10
