import numpy as np


def uq_accuracy(uq, error, percentile=0.1):
    """Shows intersection of worst by error/uq in percentile"""
    k = int(len(uq)*percentile)
    worst_uq = np.argsort(np.ravel(uq))[-k:]
    worst_error = np.argsort(np.ravel(error))[-k:]
    return len(set(worst_uq).intersection(set(worst_error)))/k
