import numpy as np
from processEmail import get_vocab_list


def emailFeatures(processedEmail):
    n = len(get_vocab_list())
    X = np.zeros((n, 1))
    for word in processedEmail:
        X[word] = 1
    return X
