import matplotlib.pyplot as plt
import numpy as np

def plotData():
    plt.plotfile('data/ex1data1.txt', delimiter=',', cols=(0, 1), 
             names=('Population Size (10,000s)', 'Profit (10,000s)'), marker='x', plotfuncs={1: 'scatter'})
    plt.show()


def loadData():
    sampleData = np.loadtxt(fname='data/ex1data1.txt', dtype= float, delimiter=',')
    #print(sampleData)
    plt.scatter(x=sampleData[:,0],y=sampleData[:,1],marker='x')
    plt.show()

def getRows():
    sampleData = np.loadtxt(fname='data/ex1data1.txt', dtype= float, delimiter=',')
    return np.size(sampleData,axis=0)

if __name__=='__main__':
    # plotData()
    loadData()
