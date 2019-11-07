import torch
from scipy.special import softmax as softmax
import numpy as np


class MCDUEBald:
    """
    Estimate uncertainty for samples with MCDUE approach
    """

    def __init__(self, net, nn_runs=25, dropout_rate=.5, num_classes=1):
        self.net = net
        self.nn_runs = nn_runs
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

    def estimate(self, X_pool, *args):
        mcd_realizations = np.zeros((X_pool.shape[0], self.nn_runs, self.num_classes))

        with torch.no_grad():
            for nn_run in range(self.nn_runs):
                prediction = self.net(X_pool, dropout_rate=self.dropout_rate)
                mcd_realizations[:, nn_run] = prediction.to('cpu')

        return self._bald(mcd_realizations)

        # mcd_realizations[:, nn_run] = np.ravel(prediction.to('cpu'))
        # return np.ravel(np.std(mcd_realizations, axis=1))

    def _entropy(self, x):
        return np.sum(-x*np.log(np.clip(x, 1e-6, 1)), axis=-1)

    def _bald(self, logits):
        # logits_samples = np.stack(
        #         [model.predict(images) for _ in range(opts.predictions_per_example)],
        #         axis=1)  # shape: [batch_size, num_samples, num_classes]

        predictions = softmax(logits, axis=-1)

        expected_entropy = np.mean(self._entropy(predictions), axis=1)
        predictive_entropy = self._entropy(np.mean(predictions, axis=1))

        return predictive_entropy - expected_entropy
