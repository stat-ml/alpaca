import numpy as np


class IdentityOracle:
    """
    Basic oracle

    Getting points for y suppose to be expensive operation
    But for this oracle it's not and it's mostly placeholder
    for more advanced oracle implementations
    """
    def __init__(self, y_set):
        self.y_set = y_set

    def evaluate(self, x_set, indices):
        """x_set and y_set are expected to be numpy ndarrays"""
        y_selected = self.y_set[indices]
        self.y_set = np.delete(self.y_set, indices, axis=0)
        return y_selected