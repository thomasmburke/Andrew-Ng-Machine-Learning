import numpy as np
import matplotlib.pyplot as plt
# Machine Learning Online Class - Exercise 1: Linear Regression
"""
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
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
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
"""
# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m
print('5x5 Identity Matrix: ')
from warmUpExercise import identity
print(identity(dimension=5))

# ======================= Part 2: Plotting =======================
print('Plotting Data ...')
from plotData import plotData, getRows
#plotData()
m=getRows()
print(m)

# =================== Part 3: Cost and Gradient descent ===================
sampleData = np.loadtxt(fname='data/ex1data1.txt', dtype= float, delimiter=',')
oneColumn = np.ones(shape=(m,1))
print('ensure the shapes of the matrixes match, o/w it will be (97,1) and (97,)')
X = np.hstack(tup=(np.ones(shape=(m,1)), sampleData[:,0].reshape(m,1)))
y = sampleData[:,1].reshape(m,1)
# Initialize fitting params
theta = np.zeros(shape=(2,1))
#print('theta: {}'.format(theta))
#print('theta,T shape: {}'.format(theta.T.shape))
#print('X shape: {}'.format(X.shape))
# Some gradient descent settings
iterations = 1500
alpha = .01
print('Testing Cost function...')
# Compute and display initial cost
from computeCost import compute_cost, slow_compute_cost
J = compute_cost(X, y, theta)
print(J)
J = slow_compute_cost(X,y,theta)
print(J)
print('Expected cost value (approx) 32.07')
J = compute_cost(X,y,np.array([[-1],[2]]))
print(J)
print('Expected cost value (approx) 54.24')
print('Running Gradient Descent...')
from gradientDescent import gradient_descent
j_history, theta = gradient_descent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent: {}'.format(theta))
print('Expected theta values (approx):')
print(' -3.6303\n  1.1664')

# Plot the linear fit
plt.scatter(x=X[:,1], y=np.dot(X,theta), marker='o')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]) @ theta
print('For population = 35,000, we predict a profit of')
print(predict1*10000)
predict2 = np.array([1, 7]) @ theta
print('For population = 70,000, we predict a profit of')
print(predict2*10000)
"""
%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
"""
