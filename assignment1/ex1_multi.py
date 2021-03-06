import numpy as np
import matplotlib.pyplot as plt
"""
%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%% ================ Part 1: Feature Normalization ================
"""
print('Loading data ...')
# Load Data
data = np.loadtxt(fname='data/ex1data2.txt', dtype=float, delimiter=',')
X = data[:, 0:2]
y = data[:, 2].reshape((len(data),1))
m = len(y)
# Print out some data points
print('First 10 examples from the dataset: \n')
print(' x = {0}, \n y = {1} \n'.format(X[0:9,:], y[0:9,:]))

# Scale features and set them to zero mean
print('Normalizing Features ...\n')
from featureNormalize import feature_normalize, feature_normalize_stddev, normalize_single_feature
X = feature_normalize_stddev(X)

# Add intercept term to X
X = np.hstack(tup=(np.ones(shape=(m,1)), X))

"""
# ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: At prediction, make sure you do the same feature normalization.
%
"""

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.03
iterations = 40000

from gradientDescent import gradient_descent
# Init Theta and Run Gradient Descent 
theta = np.zeros(shape=(3, 1))
J_history, theta = gradient_descent(X, y, theta, alpha, iterations)
print('j_history:\n{}'.format(J_history))
print('theta:\n{}'.format(theta))

# Estimate the price of a 1650 sq-ft, 3 br house
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
inputs = np.array([1650,3])
inputs = normalize_single_feature(data[:, 0:2], inputs)
inputs = np.append(np.ones(1), inputs)
print(inputs)
price = inputs @ theta # these prediction values need to be normalized
print('Predicted price of a 1650 sq-ft, 3 br house using gradient descent:\n {}'.format(price))
# Create 2 subplot, 1 for each variable
fig, axes = plt.subplots(figsize=(12,4),nrows=1,ncols=2)
axes[0].scatter(data[:,0],data[:,2],color="b")
axes[0].set_xlabel("Size (Square Feet)")
axes[0].set_ylabel("Prices")
axes[0].set_title("House prices against size of house")
axes[1].scatter(data[:,1],data[:,2],color="r")
axes[1].set_xlabel("Number of bedroom")
axes[1].set_ylabel("Prices")
axes[1].set_xticks(np.arange(1,6,step=1))
axes[1].set_title("House prices against number of bedroom")
# Enhance layout
plt.tight_layout()
plt.show()



# ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n')
"""
% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%
"""
# Load Data
data = np.loadtxt(fname='data/ex1data2.txt', dtype=float, delimiter=',')
X = data[:, 0:2]
y = data[:, 2].reshape((len(data),1))
m = len(y)

# Add intercept term to X
X = np.hstack(tup=(np.ones(shape=(m,1)), X))

# Calculate the parameters from the normal equation
from normalEqn import normal_equation
theta = normal_equation(X, y)
print('normal eqn theta:\n{}'.format(theta))

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
inputs = np.array([1,1650,3])
price = inputs @ theta
print('Predicted price of a 1650 sq-ft. 3 br house(Using normal equation)...')
print(price)
