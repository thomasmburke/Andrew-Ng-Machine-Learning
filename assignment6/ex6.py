import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
"""
%% Machine Learning Online Class
%  Exercise 6 | Support Vector Machines
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%% =============== Part 1: Loading and Visualizing Data ================
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
"""

print('Loading and Visualizing Data ...')

# Load from ex6data1: 
# You will have X, y in your environment
data1 = loadmat('data/ex6data1.mat')
X = data1['X']
y = data1['y']
m, n = X.shape

# Plot training data
pos , neg= (y==1), (y==0)
#plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+", label='Admitted')
#plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10, label='Not Admitted')
#plt.xlabel("Exam 1 score")
#plt.ylabel("Exam 2 score")
#plt.legend(loc=0)
#plt.show()
"""
%% ==================== Part 2: Training Linear SVM ====================
%  The following code will train a linear SVM on the dataset and plot the
%  decision boundary learned.
%
"""
print('Training Linear SVM ...')
from sklearn.svm import SVC
classifier = SVC(C=100, kernel='linear')
classifier.fit(X,np.ravel(y))
plt.figure(figsize=(8,6))
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+",s=50)
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c="y",marker="o",s=50)
# plotting the decision boundary
# use linspace to generate 100 points between the min and max for both x1 and x2
# then use meshgrid to create a 2D linspace (grid) of points at each of the points provided by linspace
X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.xlim(0,4.5)
plt.ylim(1.5,5)
plt.show()
"""
%% =============== Part 4: Visualizing Dataset 2 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
"""

print('Loading and Visualizing Data2 ...')

# Load from ex6data2: 
# You will have X, y in your environment
data2 = loadmat('data/ex6data2.mat')
# data set has 863 rows and 2 features
X2 = data2['X']
y2 = data2['y']
m, n = X2.shape

#plot nonlinear distribution
plt.figure(figsize=(8,6))
pos, neg = (y2==1), (y2==0)
plt.scatter(X2[pos[:,0],0],X2[pos[:,0],1],c="r",marker="+",s=50)
plt.scatter(X2[neg[:,0],0],X2[neg[:,0],1],c="y",marker="o",s=50)
#plt.show()

"""
%% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
%  After you have implemented the kernel, we can now use it to train the 
%  SVM classifier.
% 
"""

print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...')

# SVM with a guassian kernel
# rbf = Radial basis function = gaussian kernel
# by default gamma = 'auto' which is 1/n for our use we will use 30
# gamma is effectively equal to 1/sigma
classifier2 = SVC(C=1, kernel='rbf',gamma=30)
classifier2.fit(X2, np.ravel(y2))
X_1,X_2 = np.meshgrid(np.linspace(X2[:,0].min(),X2[:,0].max(),num=100),np.linspace(X2[:,1].min(),X2[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier2.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.show()
"""
%% =============== Part 6: Visualizing Dataset 3 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
%
"""
print('Loading and Visualizing Data ...')

# Load from ex6data3: 
# You will have X, y in your environment
data3 = loadmat('data/ex6data3.mat')
X3 = data3['X']
y3 = data3['y']
X3val = data3['Xval']
y3val = data3['yval']
m, n = X.shape
plt.figure(figsize=(8,6))
pos, neg = (y3==1), (y3==0)
plt.scatter(X3[pos[:,0],0],X3[pos[:,0],1],c="r",marker="+",s=50)
plt.scatter(X3[neg[:,0],0],X3[neg[:,0],1],c="y",marker="o",s=50)

"""
%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

%  This is a different dataset that you can use to experiment with. Try
%  different values of C and sigma here.
% 
"""
from dataset3Params import dataset3Params
vals = [.01, .03, .1, .3, 1, 3, 10, 30]
C, gamma = dataset3Params(X3, y3, X3val, y3val, vals)

classifier3 = SVC(C=C, gamma=gamma)
classifier3.fit(X3, np.ravel(y3))
X_1,X_2 = np.meshgrid(np.linspace(X3[:,0].min(),X3[:,0].max(),num=100),np.linspace(X3[:,1].min(),X3[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier3.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.show()
