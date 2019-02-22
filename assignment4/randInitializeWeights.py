import numpy as np


def randInitializeWeights(L_in, L_out):
    """
    Initialize random matrix of size L_outxL_in+1
    """
    epsilon = (6**0.5) / ((L_in + L_out)**0.5)
    W = np.random.rand(L_out, L_in+1) * 2 * epsilon - epsilon
    return W
