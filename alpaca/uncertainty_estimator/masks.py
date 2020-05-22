from collections import defaultdict

import torch
import numpy as np
import numpy.linalg as la
from scipy.special import softmax
from dppy.finite_dpps import FiniteDPP


DEFAULT_MASKS = ['mc_dropout', 'decorrelating_sc', 'dpp', 'k_dpp', 'ht_dpp', 'ht_k_dpp']


# It's better to use this function to get the mask then call them directly
def build_masks(names=None, **kwargs):
    masks = {
        'basic_bern': BasicBernoulliMask(),
        'mc_dropout': BasicBernoulliMask(),
        'decorrelating': DecorrelationMask(),
        'decorrelating_sc': DecorrelationMask(scaling=True),
        'ht_decorrelating': DecorrelationMask(scaling=True, ht_norm=True),
        'dpp': DPPMask(),
        'k_dpp': KDPPMask(),
        'k_dpp_noisereg': KDPPMask(noise_level=kwargs.get('noise_level', 1e-2)),
        'ht_dpp': DPPMask(ht_norm=True),
        'ht_k_dpp': KDPPMask(ht_norm=True)
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

        noise = torch.zeros(x.shape[-1]).to(x.device)
        # noise = self._make_noise(x)
        if p > 0:
            noise.bernoulli_(p).div_(p)
        # noise = noise.expand_as(x)
        return noise

    # @staticmethod
    # def _make_noise(input):
    #     return input.new().resize_as_(input)


def mc_probability(prob, k, repeats=1000):
    results = np.zeros((repeats, len(prob)))
    for i in range(repeats):
        ids = np.random.choice(len(prob), k, p=prob, replace=False)
        results[i, ids] = 1
    return np.sum(results, axis=0) / repeats


class DecorrelationMask:
    def __init__(self, scaling=False, dry_run=True, ht_norm=False):
        self.layer_correlations = {}
        self.scaling = scaling  # use adaptive scaling before softmax
        self.dry_run = dry_run
        self.ht_norm = ht_norm
        self.norm = {}

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

            # if self.ht_norm:
            if self.ht_norm:
                k = int(x.shape[-1]*(1-dropout_rate))
                probabilities = self.layer_correlations[layer_num]
                samples = max(1000, 4*x.shape[-1])
                self.norm[layer_num] = np.reciprocal(mc_probability(probabilities, k, samples))
            # Initially we should pass identity mask,
            # otherwise we won't get right correlations for all layers
            return x.data.new(x.data.size()[-2]).fill_(1)

        mask_len = x.data.size()[-1]
        mask = torch.zeros(mask_len).double().to(x.device)
        k = int(mask_len*(1-dropout_rate))
        ids = np.random.choice(len(mask), k, p=self.layer_correlations[layer_num], replace=False)

        if self.ht_norm:
            mask[ids] = torch.DoubleTensor(self.norm[layer_num][ids])
        else:
            mask[ids] = 1 / (1 - dropout_rate)

        return x.data.new(mask)

    def reset(self):
        self.layer_correlations = {}


ATTEMPTS = 30


class DPPMask:
    def __init__(self, ht_norm=False):
        self.dpps = {}
        self.layer_correlations = {}  # keep for debug purposes
        # Flag for uncertainty estimator to make first run without taking the result
        self.dry_run = True

        self.ht_norm = ht_norm
        self.norm = {}

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        if layer_num not in self.layer_correlations:
            # warm-up, generatign correlations masks
            x_matrix = x.cpu().numpy()

            self.x_matrix = x_matrix
            micro = 1e-12
            x_matrix += np.random.random(x_matrix.shape) * micro  # for computational stability
            correlations = np.corrcoef(x_matrix.T)
            self.dpps[layer_num] = FiniteDPP('likelihood', **{'L': correlations})
            self.layer_correlations[layer_num] = correlations

            if self.ht_norm:
                L = torch.DoubleTensor(correlations).cuda()
                E = torch.eye(len(correlations)).cuda()
                K = torch.mm(L, torch.inverse(L + E))

                self.norm[layer_num] = torch.reciprocal(torch.diag(K))  # / len(correlations)
                self.L = L
                self.K = K

            return x.data.new(x.data.size()[-1]).fill_(1)

        # sampling nodes ids
        dpp = self.dpps[layer_num]

        for _ in range(ATTEMPTS):
            dpp.sample_exact()
            ids = dpp.list_of_samples[-1]
            if len(ids):  # We should retry if mask is zero-length
                break

        mask_len = x.data.size()[-1]
        mask = torch.zeros(mask_len).double().cuda()
        if self.ht_norm:
            mask[ids] = self.norm[layer_num][ids]
        else:
            mask[ids] = mask_len / len(ids)

        return x.data.new(mask)

    def reset(self):
        self.layer_correlations = {}


class KDPPMask:
    def __init__(self, noise_level=None, tol_level=1e-3, ht_norm=False):
        self.layer_correlations = {}
        self.dry_run = True
        self.dpps = {}
        self.ranks = {}
        self.ranks_history = defaultdict(list)
        self.noise_level = noise_level
        self.tol_level = tol_level

        self.ht_norm = ht_norm
        self.norm = {}

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

            if self.ht_norm:
                L = torch.DoubleTensor(correlations).cuda()
                E = torch.eye(len(correlations)).cuda()
                K = torch.mm(L, torch.inverse(L + E))

                self.norm[layer_num] = torch.reciprocal(torch.diag(K))  # / len(correlations)
                self.L = L
                self.K = K

            return x.data.new(x.data.size()[-1]).fill_(1)

        mask_len = x.size()[-1]
        mask = torch.zeros(mask_len).double().cuda()
        k = int(self.ranks[layer_num] * (1 - dropout_rate))
        ids = self.dpps[layer_num].sample_exact_k_dpp(k)

        if self.ht_norm:
            mask[ids] = self.norm[layer_num][ids]
        else:
            mask[ids] = mask_len / len(ids)
            # mask[ids] = 1 / (1 - dropout_rate + 1e-10)

        return x.data.new(mask)

    def reset(self):
        self.layer_correlations = {}
