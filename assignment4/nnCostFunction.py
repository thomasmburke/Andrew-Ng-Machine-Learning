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
    a1 = X
    z2 = a1 @ Theta1.T
    a2 = sigmoid(z2)
    a2 = np.hstack(tup=(np.ones(shape=(m,1)),a2))
    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)
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
        J = J + sum(-y10[:,i] * np.log(a3[:,i]) - (1 - y10[:,i]) * np.log(1 - a3[:,i]))
    # cost without regularization
    cost = J = 1/m * J
    # Now to compute the cost with regularization
    J_reg = np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:]))
    J_reg = lambdaValue / (2 * m) * J_reg
    J_reg = J + J_reg
    
    #Implement back propogation to compute gradients
    grad1 = np.zeros((Theta1.shape))
    grad2 = np.zeros((Theta2.shape))
    for i in range(m):
        a1i = a1[i,:] # [401,]
        a2i = a2[i,:] # [26,]
        a3i = a3[i, :] # [10,]
        d3 = a3i - y10[i,:] # [10,]
        d2 = Theta2.T @ d3.T * sigmoid_gradient(np.hstack((1,a1i @ Theta1.T))) # [26,]
        # index d2[1:] to not include the bias unit
        grad1= grad1 + d2[1:,np.newaxis] @ a1i[:,np.newaxis].T # 25x1 @ 1x401 
        grad2 = grad2 + d3[:,np.newaxis] @ a2i[:,np.newaxis].T # 10x1 @ 1x26
        #print(d3.T[:,np.newaxis].shape)

    grad1 = 1/m * grad1
    grad2 = 1/m * grad2

    grad1_reg = grad1 + (lambdaValue/m) * np.hstack((np.zeros((Theta1.shape[0],1)),Theta1[:,1:]))
    grad2_reg = grad2 + (lambdaValue/m) * np.hstack((np.zeros((Theta2.shape[0],1)),Theta2[:,1:]))

    return cost, grad1, grad2,J_reg, grad1_reg,grad2_reg
