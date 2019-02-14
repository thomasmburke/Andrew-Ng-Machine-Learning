import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return (1/(1+np.exp(-z)))

if __name__=='__main__':
    print(sigmoid(0))
