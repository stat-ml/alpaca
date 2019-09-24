import torch
import numpy as np


class NNGPRegression:
    def __init__(self, net, nn_runs=25, diag_eps=1e-6, dropout_rate=0.5):
        self.net = net
        self.nn_runs = nn_runs
        self.diag_eps = diag_eps
        self.dropout_rate = dropout_rate

    def estimate(self, x_pool, x_train):
        train_pool_samples = np.concatenate([x_train, x_pool])
        train_len = len(x_train)

        mcd_predictions = self._mcd_predict(train_pool_samples)

        # covariance matrix with regularization
        cov_matrix_train = np.cov(mcd_predictions[:train_len, :], ddof=0)
        cov_matrix_inv = np.linalg.inv(cov_matrix_train + np.eye(train_len)*self.diag_eps)

        pool_samples = mcd_predictions[train_len:]
        Qs = self.simple_covs(mcd_predictions[:train_len, :], pool_samples).T
        Qt_K_Q = np.matmul(np.matmul(Qs.T, cov_matrix_inv), Qs)
        KKs = np.var(pool_samples, axis=1)

        ws = KKs - np.diag(Qt_K_Q)

        gp_ue = [0 if w < 0 else np.sqrt(w) for w in np.ravel(ws)]
        return np.ravel(gp_ue)

    def _mcd_predict(self, train_pool_samples):
        mcd_predictions = np.zeros((train_pool_samples.shape[0], self.nn_runs))
        with torch.no_grad():
            for nn_run in range(self.nn_runs):
                prediction = self.net(train_pool_samples, dropout_rate=self.dropout_rate).to('cpu')
                mcd_predictions[:, nn_run] = np.ravel(prediction)
        return mcd_predictions

    @staticmethod
    def simple_covs(a, b):
        ac = a - a.mean(axis=-1, keepdims=True)
        bc = (b - b.mean(axis=-1, keepdims=True)) / b.shape[-1]
        return np.dot(ac, bc.T).T

