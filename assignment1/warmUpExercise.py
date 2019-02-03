import numpy as np

def identity(dimension):
    """
    Summary: return identity matrix of size dimension
    """
    return np.eye(dimension)

if __name__=='__main__':
    print(identity(5))
