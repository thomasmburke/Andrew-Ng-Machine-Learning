import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, y):
    pos , neg= (y==1), (y==0)
    plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
    plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(["Admitted","Not admitted"],loc=0)
    plt.show()
