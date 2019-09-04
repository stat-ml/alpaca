import numpy as np
import torch


class MCDUE:
    """
    Estimate uncertainty for samples with MCDUE approach
    """
    def __init__(self, net, nn_runs=25, dropout_rate=.5):
        self.net = net
        self.nn_runs = nn_runs
        self.dropout_rate = dropout_rate

    def estimate(self, X_pool, X_train, y_train):
        mcd_realizations = np.zeros((X_pool.shape[0], self.nn_runs))

        with torch.no_grad():
            for nn_run in range(self.nn_runs):
                prediction = self.net(X_pool, dropout_rate=self.dropout_rate)
                mcd_realizations[:, nn_run] = np.ravel(prediction.to('cpu'))

        return np.ravel(np.std(mcd_realizations, axis=1))


class MCDUEMasked:
    """
    Estimate uncertainty for samples with MCDUE approach
    """
    def __init__(self, net, nn_runs=25, dropout_rate=.5, dropout_mask=None):
        self.net = net
        self.nn_runs = nn_runs
        self.dropout_rate = dropout_rate
        self.dropout_mask = dropout_mask

    def estimate(self, X_pool, X_train, y_train):
        mcd_realizations = np.zeros((X_pool.shape[0], self.nn_runs))

        with torch.no_grad():
            # Some mask needs first run without dropout, i.e. decorrelation mask
            if hasattr(self.dropout_mask, 'dry_run') and self.dropout_mask.dry_run:
                self.net(X_pool, dropout_rate=self.dropout_rate, dropout_mask=self.dropout_mask)

            # Get mcdue estimation
            for nn_run in range(self.nn_runs):
                prediction = self.net(
                    X_pool, dropout_rate=self.dropout_rate, dropout_mask=self.dropout_mask
                ).to('cpu')
                mcd_realizations[:, nn_run] = np.ravel(prediction)

        return np.ravel(np.std(mcd_realizations, axis=1))

