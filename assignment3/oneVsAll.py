import numpy as np
from lrCostFunction import lrCostFunction

def gradientDescent(X,y,theta,alpha,num_iters,Lambda):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha

    return theta and the list of the cost of theta during each iteration
    """

    m=len(y)
    J_history =[]

    for i in range(num_iters):
        cost, grad = lrCostFunction(theta,X,y,Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)

    return theta , J_history


def oneVsAll(X, y, num_labels, Lambda):
    """
    Takes in numpy array of X,y, int num_labels and float lambda to train multiple logistic regression classifiers
    depending on the number of num_labels using gradient descent.

    Returns a matrix of theta, where the i-th row corresponds to the classifier for label i
    """
    m, n = X.shape
    # initialize a theta column with all features + the bias factor
    initial_theta = np.zeros((n+1,1))
    all_theta = []
    all_J=[]
    # add intercept terms
    X = np.hstack((np.ones((m,1)),X))
    for i in range(1,num_labels+1):
        # print(np.where(y==i,1,0)) # This shows y reconstructed with only 1 and 0s for when y matches i it will be 1
        theta , J_history = gradientDescent(X,np.where(y==i,1,0),initial_theta,1,300,Lambda)
        # each loop adds 401 theta values to the list because n=400 pixels worth of dat plus the bias
        all_theta.extend(theta)
        all_J.extend(J_history)
    return np.array(all_theta).reshape(num_labels,n+1), all_J
