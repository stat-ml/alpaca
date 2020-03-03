from collections import defaultdict

import torch
import numpy as np
import numpy.linalg as la
from scipy.special import softmax
from dppy.finite_dpps import FiniteDPP


# DEFAULT_MASKS = ['basic_bern', 'decorrelating_sc', 'dpp', 'k_dpp', 'k_dpp_noisereg']
DEFAULT_MASKS = ['mc_dropout', 'decorrelating_sc', 'dpp', 'k_dpp']


# It's better to use this function to get the mask then call them directly
def build_masks(names=None, **kwargs):
    masks = {
        'basic_bern': BasicBernoulliMask(),
        'mc_dropout': BasicBernoulliMask(),
        'decorrelating': DecorrelationMask(),
        'decorrelating_sc': DecorrelationMask(scaling=True, dry_run=False),
        'dpp': DPPMask(),
        'k_dpp': KDPPMask(),
        'k_dpp_noisereg': KDPPMask(noise_level=kwargs.get('noise_level', 1e-2))
    }
    if names is None:
        return masks
    return {name: masks[name] for name in names}


# Build single mask
def build_mask(name, **kwargs):
    return build_masks([name], **kwargs)[name]


class BasicBernoulliMask:
    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        p = 1 - dropout_rate

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        noise = self._make_noise(x)
        if p == 0:
            noise.fill_(0)
        else:
            noise.bernoulli_(p).div_(p)

        noise = noise.expand_as(x)
        return noise

    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)


class DecorrelationMask:
    def __init__(self, scaling=False, dry_run=True):
        self.layer_correlations = {}
        self.scaling = scaling  # use adaptive scaling before softmax
        self.dry_run = dry_run

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        if layer_num not in self.layer_correlations:
            x_matrix = x.cpu().numpy()
            self.x_matrix = x_matrix

            noise = 1e-8 * np.random.rand(*x_matrix.shape)  # to prevent degeneration
            corrs = np.sum(np.abs(np.corrcoef((x_matrix+noise).T)), axis=1)
            scores = np.reciprocal(corrs)
            if self.scaling:
                scores = 4 * scores / max(scores)
            self.layer_correlations[layer_num] = softmax(scores)
            # Initially we should pass identity mask,
            # otherwise we won't get right correlations for all layers
            return x.data.new(x.data.size()[-1]).fill_(1)
        mask = x.data.new(x.data.size()[-1]).fill_(0)
        k = int(len(mask)*(1-dropout_rate))
        ids = np.random.choice(len(mask), k, p=self.layer_correlations[layer_num])
        mask[ids] = 1 / (1 - dropout_rate + 1e-10)

        return x.data.new(mask)

    def reset(self):
        self.layer_correlations = {}


ATTEMPTS = 10


class DPPMask:
    def __init__(self):
        self.dpps = {}
        self.layer_correlations = {}  # keep for debug purposes
        # Flag for uncertainty estimator to make first run without taking the result
        self.dry_run = True

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        if layer_num not in self.layer_correlations:
            # warm-up, generatign correlations masks
            x_matrix = x.cpu().numpy()

            self.x_matrix = x_matrix
            correlations = np.corrcoef(x_matrix.T)
            self.dpps[layer_num] = FiniteDPP('likelihood', **{'L': correlations})
            self.layer_correlations[layer_num] = correlations

            return x.data.new(x.data.size()[-1]).fill_(1)

        # sampling nodes ids
        dpp = self.dpps[layer_num]

        for _ in range(ATTEMPTS):
            dpp.sample_exact()
            ids = dpp.list_of_samples[-1]
            if len(ids):  # We should retry if mask is zero-length
                break

        mask_len = x.data.size()[-1]
        mask = x.data.new(mask_len).fill_(0)
        mask[ids] = mask_len/len(ids)

        return x.data.new(mask)

    def reset(self):
        self.layer_correlations = {}


class KDPPMask:
    def __init__(self, noise_level=None, tol_level=1e-3):
        self.layer_correlations = {}
        self.dry_run = True
        self.dpps = {}
        self.ranks = {}
        self.ranks_history = defaultdict(list)
        self.noise_level = noise_level
        self.tol_level = tol_level

    def _rank(self, dpp):
        N = dpp.eig_vecs.shape[0]
        tol = max(np.max(dpp.L_eig_vals) * N * np.finfo(np.float).eps, self.tol_level)
        rank = np.count_nonzero(dpp.L_eig_vals > tol)
        return rank

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        if layer_num not in self.layer_correlations:
            x_matrix = x.cpu().numpy()

            correlations = np.corrcoef(x_matrix.T)
            if self.noise_level is not None:
                correlations += self.noise_level * np.eye(len(correlations))
            self.dpps[layer_num] = FiniteDPP('likelihood', **{'L': correlations})
            self.dpps[layer_num].sample_exact()  # to trigger eig values generation
            self.ranks[layer_num] = self._rank(self.dpps[layer_num])

            # Keep data for debugging
            self.ranks_history[layer_num].append(self.ranks[layer_num])
            self.layer_correlations[layer_num] = correlations

            return x.data.new(x.data.size()[-1]).fill_(1)

        mask = x.data.new(x.data.size()[-1]).fill_(0)
        k = int(self.ranks[layer_num] * (1 - dropout_rate))
        self.dpps[layer_num].sample_exact_k_dpp(k)

        ids = self.dpps[layer_num].list_of_samples[-1]

        mask[ids] = 1 / (1 - dropout_rate + 1e-10)

        return x.data.new(mask)

    def reset(self):
        self.layer_correlations = {}
