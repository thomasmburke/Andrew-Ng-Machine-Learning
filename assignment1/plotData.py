import matplotlib.pyplot as plt
import numpy as np

def plotData():
    plt.plotfile('data/ex1data1.txt', delimiter=',', cols=(0, 1), 
             names=('Size (ft^2)', 'Price'), marker='x', plotfuncs={1: 'scatter'})
    plt.show()


def loadData():
    sampleData = np.loadtxt(fname='data/ex1data1.txt', dtype= float, delimiter=',')
    #print(sampleData)
    plt.scatter(x=sampleData[:,0],y=sampleData[:,1],marker='x')
    plt.show()

if __name__=='__main__':
    # plotData()
    loadData()
