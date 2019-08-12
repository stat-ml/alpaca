import numpy as np


class MCDUE:
    """
    Estimate uncertainty for samples with MCDUE approach
    """
    def __init__(self, net, nn_runs=25, probability=.5):
        self.net = net
        self.nn_runs = nn_runs
        self.probability = probability

    def estimate(self, X_pool, X_train, y_train):
        mcd_realizations = np.zeros((X_pool.shape[0], self.nn_runs))

        for nn_run in range(self.nn_runs):
            prediction = self.net.predict(X_pool, dropout_rate=self.probability)
            mcd_realizations[:, nn_run] = np.ravel(prediction)

        return np.ravel(np.std(mcd_realizations, axis=1))
