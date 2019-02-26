import numpy as np
import scipy.io
"""
%% Machine Learning Online Class
%  Exercise 5 | Regularized Linear Regression and Bias-Variance
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     linearRegCostFunction.m
%     learningCurve.m
%     validationCurve.m
%
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
%
"""
# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment

data = scipy.io.loadmat('data/ex5data1.mat')
X = data['X']
y = data['y']
Xtest = data['Xtest']
ytest = data['ytest']
Xval = data['Xval']
yval = data['yval']
m, n = X.shape
import scipy.stats
X_description = scipy.stats.describe(X)
print(X_description)
# Plot training data
"""
%% =========== Part 2: Regularized Linear Regression Cost =============
%  You should now implement the cost function for regularized linear 
%  regression. 
%
"""

theta = np.array([[1], [1]])
from linearRegCostFunction import linearRegCostFunction
J = linearRegCostFunction(X, y, theta, 1)
print('Cost at theta = [1 ; 1]: {} \n(this value should be about 303.993192)\n'.format(J))
"""
%% =========== Part 3: Regularized Linear Regression Gradient =============
%  You should now implement the gradient for regularized linear 
%  regression.
%
"""
J, grad = linearRegCostFunction(X, y, theta, 1)
print('Gradient at theta = [1 ; 1]: {0}, {1}\n(this value should be about [-15.303016; 598.250744])\n'.format(grad[0], grad[1]))
"""
%% =========== Part 4: Train Linear Regression =============
%  Once you have implemented the cost and gradient correctly, the
%  trainLinearReg function will use your cost function to train 
%  regularized linear regression.
% 
%  Write Up Note: The data is non-linear, so this will not give a great 
%                 fit.
%
"""
#  Train linear regression with lambda = 0
from trainLinearReg import trainLinearReg
theta, J_Hist = trainLinearReg(X, y, 0)
print('theta\n')
print(theta)
#print('J_Hist:\n')
#print(J_Hist)

import matplotlib.pyplot as plt
plt.plot(J_Hist)
plt.xlabel('num iterations')
plt.ylabel('cost or J(THETA)')
plt.show()
plt.scatter(X, y, marker='x', c='blue')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
x_value = np.linspace(start=np.min(X),stop=np.max(X))
y_value = x_value*theta[1]+theta[0]
#x_value=[x for x in range(-50,40)]
#y_value=[y_hat*theta[1]+theta[0] for y_hat in x_value]
plt.plot(x_value,y_value,color="r")
plt.ylim(-5,40)
plt.xlim(-50,40)
plt.show()
"""
%% =========== Part 5: Learning Curve for Linear Regression =============
%  Next, you should implement the learningCurve function. 
%
%  Write Up Note: Since the model is underfitting the data, we expect to
%                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
%
"""
from learningCurve import learningCurve
lambdaValue = 0
error_train, error_val =  learningCurve(X, y, Xval, yval, lambdaValue)
plt.plot(range(12),error_train,label="Train")
plt.plot(range(12),error_val,label="Cross Validation",color="r")
plt.title("Learning Curve for Linear Regression")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()
plt.show()
"""
%% =========== Part 6: Feature Mapping for Polynomial Regression =============
%  One solution to this is to use polynomial regression. You should now
%  complete polyFeatures to map each example into its powers
"""
from polyFeatures import polyFeatures
p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_poly=sc_X.fit_transform(X_poly)
#X_poly = np.hstack((np.ones((X_poly.shape[0],1)),X_poly))
# Map Xtest onto polynomial features and normalize
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = sc_X.transform(X_poly_test)
# Map Xval onto polynomial features and normalize
X_poly_val = polyFeatures(Xval, p)
X_poly_val = sc_X.transform(X_poly_val)

theta_poly, J_Hist = trainLinearReg(X_poly, y, 0, .3, 20000)
plt.scatter(X,y,marker="x",color="r")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
x_value=np.linspace(-55,65,2400)

# Map the X values and normalize
x_value_poly = polyFeatures(x_value[:,np.newaxis], p)
x_value_poly = sc_X.transform(x_value_poly)
x_value_poly = np.hstack((np.ones((x_value_poly.shape[0],1)),x_value_poly))
y_value= x_value_poly @ theta_poly
plt.plot(x_value,y_value,"--",color="b")
plt.show()

"""
%% =========== Part 7: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with 
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%
"""

lambdaValue = 0
theta = trainLinearReg(X_poly, y, lambdaValue)
"""
%% =========== Part 8: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of 
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end
"""
