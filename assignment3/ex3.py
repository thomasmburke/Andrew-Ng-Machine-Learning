import numpy as np
"""
%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
"""
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.


# Load Training Data
print('Loading and Visualizing Data ...\n')
import scipy.io
data = scipy.io.loadmat('data/ex3data1.mat') # training data stored in arrays X, y
X = data['X']
y = data['y']
print('size of X: {}'.format(X.shape))
print('size of y: {}'.format(y.shape))
m = len(X)
print(m)
print(np.around(X[1,:], decimals=2))
# Randomly select 100 data points to display
sample = X[np.random.choice(X.shape[0], 100, replace=False), :]
#print(sample)
#print(sample.shape)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
    for j in range(10):
        axis[i,j].imshow(X[np.random.randint(0,5001),:].reshape(20,20,order="F"), cmap="hot") #reshape back to 20 pixel by 20 pixel
        axis[i,j].axis("off")
plt.show()
"""
%% ============ Part 2a: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%
"""

# Test case for lrCostFunction
print('Testing lrCostFunction() with regularization')
from lrCostFunction import lrCostFunction
theta_t = np.array([-2,-1,1,2]).reshape(4,1)
X_t =np.array([np.linspace(0.1,1.5,15)]).reshape(3,5).T
X_t = np.hstack((np.ones((5,1)), X_t))
y_t = np.array([1,0,1,0,1]).reshape(5,1)
J, grad = lrCostFunction(theta_t, X_t, y_t, 3)
print("Cost:",J,"Expected cost: 2.534819")
print("Gradients:\n",grad,"\nExpected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003")


print(y)

# ============ Part 2b: One-vs-All Training ============
print('Training One-vs-All Logistic Regression...')
from oneVsAll import oneVsAll
all_theta, all_J = oneVsAll(X, y, num_labels, 0.1)
plt.plot(all_J[0:300])
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()


# ================ Part 3: Predict for One-Vs-All ================
from predictOneVsAll import predictOneVsAll
pred = predictOneVsAll(all_theta, X)
print(pred)
print("Training Set Accuracy:",sum(pred[:,np.newaxis]==y)[0]/5000*100,"%")
