import numpy as np


class RandomEstimator:
    def estimate(self, X_pool, *args):
        return np.ones(X_pool.shape[0])
