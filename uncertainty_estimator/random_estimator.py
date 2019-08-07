import numpy as np


class RandomEstimator:
    def estimate(self, X_pool, X_train, y_train):
        return np.zeros(X_pool.shape[0])
