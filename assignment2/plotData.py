import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, y, theta):
    pos , neg= (y==1), (y==0)
    plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+", label='Admitted')
    plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10, label='Not Admitted')
    #x_value= np.array([np.min(X[:,1]),np.max(X[:,1])])
    #y_value=-(theta[0] +theta[1]*x_value)/theta[2]
    x_value = np.linspace(start=np.min(X[:,1]), stop=(np.max(X[:,1])))
    y_value = -(theta[1]/theta[2]) * x_value - theta[0] / theta[2]
    plt.plot(x_value,y_value, "r")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(loc=0)
    plt.show()
