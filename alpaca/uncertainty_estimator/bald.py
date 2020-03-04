import torch
from scipy.special import softmax
import numpy as np
from .masks import build_mask


class Bald:
    """
    Estimate uncertainty in classification tasks for samples with MCDUE approach
    """
    def __init__(self, net, nn_runs=25, dropout_rate=.5, num_classes=1):
        self.net = net
        self.nn_runs = nn_runs
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

    def estimate(self, X_pool, *args):
        mcd_runs = np.zeros((X_pool.shape[0], self.nn_runs, self.num_classes))

        with torch.no_grad():
            for nn_run in range(self.nn_runs):
                prediction = self.net(X_pool, dropout_rate=self.dropout_rate)
                mcd_runs[:, nn_run] = prediction.to('cpu')

        return _bald(mcd_runs)


class BaldMasked:
    """
    Estimate uncertainty for samples with MCDUE approach
    """
    # TODO: different acquisition to separate classes, it's not BALD
    def __init__(
            self, net, nn_runs=100, dropout_mask=None, dropout_rate=.5,
            num_classes=2, keep_runs=False, acquisition='var_ratio'):
        self.net = net
        self.nn_runs = nn_runs
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        if isinstance(dropout_mask, str):
            dropout_mask = build_mask(dropout_mask)
        self.dropout_mask = dropout_mask
        self.keep_runs = keep_runs
        self._mcd_runs = np.array([])
        self.acquisition = acquisition

    def estimate(self, X_pool, *args):
        mcd_runs = np.zeros((X_pool.shape[0], self.nn_runs, self.num_classes))

        with torch.no_grad():
            self.net.eval()
            # Some mask needs first run without dropout, i.e. decorrelation mask
            if hasattr(self.dropout_mask, 'dry_run') and self.dropout_mask.dry_run:
                self.net(X_pool, dropout_rate=self.dropout_rate, dropout_mask=self.dropout_mask)

            # Get mcdue estimation
            for nn_run in range(self.nn_runs):
                prediction = self.net(
                    X_pool, dropout_rate=self.dropout_rate, dropout_mask=self.dropout_mask
                ).to('cpu')
                mcd_runs[:, nn_run] = prediction

            if self.keep_runs:
                self._mcd_runs = mcd_runs

        return self._aquisition(mcd_runs)

    def _aquisition(self, mcd_runs):
        if self.acquisition == 'var_ratio':
            predictions = np.argmax(mcd_runs, axis=-1)
            # count how many time repeats the strongest class
            mode_count = lambda preds : np.max(np.bincount(preds))
            modes = [mode_count(point) for point in predictions]
            ue = 1 - np.array(modes) / self.nn_runs
            return ue
        elif self.acquisition == 'var_soft':
            probabilities = softmax(mcd_runs, axis=-1)
            ue = np.mean(np.std(probabilities, axis=-2), axis=-1)
            return ue
        elif self.acquisition == 'bald':
            return _bald(mcd_runs)
        else:
            raise ValueError

    def reset(self):
        if hasattr(self.dropout_mask, 'reset'):
            self.dropout_mask.reset()

    def last_mcd_runs(self):
        """Return model prediction for last uncertainty estimation"""
        if not self.keep_runs:
            print("mcd_runs: You should set `keep_runs=True` to properly use this method")
        return self._mcd_runs


class BaldEnsemble:
    """
    Estimate uncertainty in classification tasks for samples with Ensemble approach
    """
    def __init__(self, ensemble, num_classes=1):
        self.ensemble = ensemble
        self.num_classes = num_classes

    def estimate(self, x_pool, *args):
        with torch.no_grad():
            logits = np.array(self.ensemble(x_pool, reduction=None).cpu())

        return _bald(np.swapaxes(logits, 0, 1))


def _entropy(x):
    return np.sum(-x*np.log(np.clip(x, 1e-8, 1)), axis=-1)


def _bald(logits):
    predictions = softmax(logits, axis=-1)

    predictive_entropy = _entropy(np.mean(predictions, axis=1))
    expected_entropy = np.mean(_entropy(predictions), axis=1)

    return predictive_entropy - expected_entropy
