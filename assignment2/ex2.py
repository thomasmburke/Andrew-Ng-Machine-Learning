import numpy as np
import matplotlib.pyplot as plt
"""
%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
"""
# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
names = ('Exam 1 Score', 'Exam 2 Score', 'Admitted')
#data = np.genfromtxt(fname='data/ex2data1.txt', dtype=float, delimiter=',', names=names)
data = np.genfromtxt(fname='data/ex2data1.txt', dtype=float, delimiter=',')
print(type(data))
X = data[:, 0:2]
y = data[:, 2].reshape((len(data),1))
m, n = X.shape

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(fit_intercept=True, C = 1e10)
clf.fit(X, np.ravel(y))

print (clf.intercept_, clf.coef_)
# ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

#print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
#from plotData import plot_data
#plot_data(X, y)

# ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You need to complete the code in 
#  costFunction.py
#  Setup the data matrix appropriately, and add ones for the intercept term
# Add intercept term to x and X_test
X = np.hstack(tup=(np.ones(shape=(m,1)),X))

# Initialize fitting parameters
initial_theta = np.zeros(shape=(n + 1, 1))

from costFunction import compute_logistic_cost, compute_cost 
# Compute and display initial cost and gradient
cost, grad = compute_logistic_cost(initial_theta, X, y);

print('Cost at initial theta (zeros): {}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): {}'.format(grad))
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24],[0.2],[0.2]])
cost, grad = compute_logistic_cost(test_theta, X, y)

print('\nCost at test theta: {}'.format(cost))
print('Expected cost (approx): 0.218')
print('Gradient at test theta: {}'.format(grad))
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')



# ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Set options for fminunc
from featureNormalization import feature_scaling
#print(feature_scaling(X))
#from scipy.optimize import fmin_bfgs
#xopt = fmin_bfgs(lambda x: compute_logistic_cost(initial_theta,X, y)[0], initial_theta, lambda x: compute_logistic_cost(initial_theta, X, y)[1])
#print(xopt)

clf = LogisticRegression(fit_intercept=False, C = 1e10, solver='lbfgs', max_iter=400)
clf.fit(X, np.ravel(y))

print (clf.intercept_, clf.coef_)
theta = clf.coef_.T
print('theta: {}'.format(theta))
print('theta: {}'.format(type(theta)))
#print weights

print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')


print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
from plotData import plot_data
plot_data(X[:,1:], y, theta)
"""
% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
%  Furthermore, you will compute the training and test set accuracies of 
%  our model.
%
%  Your task is to complete the code in predict.m

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 

prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n'], prob);
fprintf('Expected value: 0.775 +/- 0.002\n\n');

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (approx): 89.0\n');
"""
