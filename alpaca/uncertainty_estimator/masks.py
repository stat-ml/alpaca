from collections import defaultdict

import torch
import numpy as np
from scipy.optimize import root_scalar
import numpy.linalg as la
from scipy.special import softmax
from dppy.finite_dpps import FiniteDPP


DEFAULT_MASKS = ['mc_dropout', 'ht_decorrelating', 'ht_dpp', 'ht_k_dpp', 'ht_leverages', 'cov_leverages']


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
        'ht_k_dpp': KDPPMask(ht_norm=True),
        'cov_dpp': DPPMask(ht_norm=True, covariance=True),
        'cov_k_dpp': KDPPMask(ht_norm=True, covariance=True),
        'ht_leverages': LeverageScoreMask(ht_norm=True, lambda_=1),
        'cov_leverages': LeverageScoreMask(ht_norm=True, lambda_=1, covariance=True),
    }
    if names is None:
        return masks
    return {name: masks[name] for name in names}


# Build single mask
def build_mask(name, **kwargs):
    return build_masks([name], **kwargs)[name]


class BasicBernoulliMask:
    def __init__(self):
        self.dry_run = True

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        p = 1 - dropout_rate

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        noise = torch.zeros(x.shape[-1]).double().to(x.device)
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
            mask_len = x.shape[-1]

            noise = 1e-8 * np.random.rand(*x_matrix.shape)  # to prevent degeneration
            corrs = np.sum(np.abs(np.corrcoef((x_matrix+noise).T)), axis=1)
            scores = np.reciprocal(corrs)
            if self.scaling:
                scores = 4 * scores / max(scores)
            self.layer_correlations[layer_num] = softmax(scores)

            if self.ht_norm:
                # Horvitz-Thopson normalization (1/marginal_prob for each element)
                k = int(mask_len*(1-dropout_rate))
                probabilities = self.layer_correlations[layer_num]
                samples = max(1000, 4*x.shape[-1])
                self.norm[layer_num] = np.reciprocal(mc_probability(probabilities, k, samples))
            # Initially we should pass identity mask,
            # otherwise we won't get right correlations for all layers
            mask = torch.ones(mask_len).to(x.device)
            return mask

        mask_len = x.shape[-1]
        mask = torch.zeros(mask_len).double().to(x.device)
        k = int(mask_len*(1-dropout_rate))
        ids = np.random.choice(len(mask), k, p=self.layer_correlations[layer_num], replace=False)

        if self.ht_norm:
            mask[ids] = torch.DoubleTensor(self.norm[layer_num][ids]).to(x.device)
        else:
            mask[ids] = 1 / (1 - dropout_rate)

        return mask

    def reset(self):
        self.layer_correlations = {}


class LeverageScoreMask:
    def __init__(self, dry_run=True, ht_norm=True, lambda_=1, covariance=False):
        self.layer_correlations = {}
        self.dry_run = dry_run
        self.ht_norm = ht_norm
        self.norm = {}
        self.lambda_ = lambda_
        self.covariance = covariance

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        mask_len = x.shape[-1]
        k = int(mask_len * (1 - dropout_rate))

        if layer_num not in self.layer_correlations:
            x_matrix = x.cpu().numpy()
            self.x_matrix = x_matrix
            if self.covariance:
                K = np.cov(x_matrix.T)
            else:
                K = np.corrcoef(x_matrix.T)
            I = np.eye(len(K))
            leverages_matrix = np.dot(K, np.linalg.inv(K+self.lambda_*I))
            probabilities = np.diagonal(leverages_matrix)
            probabilities = probabilities / sum(probabilities)
            self.layer_correlations[layer_num] = probabilities

            if self.ht_norm:
                # Horvitz-Thopson normalization (1/marginal_prob for each element)
                probabilities = self.layer_correlations[layer_num]
                samples = max(1000, 4*x.shape[-1])
                self.norm[layer_num] = np.reciprocal(mc_probability(probabilities, k, samples))
            # Initially we should pass identity mask,
            # otherwise we won't get right correlations for all layers
            mask = torch.ones(mask_len).to(x.device)
            return mask

        mask = torch.zeros(mask_len).double().to(x.device)
        ids = np.random.choice(len(mask), k, p=self.layer_correlations[layer_num], replace=False)

        if self.ht_norm:
            mask[ids] = torch.DoubleTensor(self.norm[layer_num][ids]).to(x.device)
        else:
            mask[ids] = 1 / (1 - dropout_rate)

        return mask

    def reset(self):
        self.layer_correlations = {}


ATTEMPTS = 30


class DPPMask:
    def __init__(self, ht_norm=False, covariance=False):
        self.dpps = {}
        self.layer_correlations = {}  # keep for debug purposes # Flag for uncertainty estimator to make first run without taking the result
        self.dry_run = True
        self.ht_norm = ht_norm
        self.norm = {}
        self.covariance = covariance

        ## For batch loaders
        self.freezed = False
        self.masks = {}

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        mask_len = x.shape[-1]
        if layer_num not in self.layer_correlations:
            # warm-up, generatign correlations masks
            x_matrix = x.cpu().numpy()
            if self.freezed:
                self.x_matrices[layer_num].append(x_matrix)
            else:
                self._setup_dpp(x_matrix, layer_num)
            return torch.ones(mask_len).to(x.device)

        if self.freezed:
            mask = self.masks[layer_num]
        else:
            mask = self._generate_mask(layer_num, mask_len)
        return mask

    def freeze(self, dry_run):
        self.freezed = True
        if dry_run:
            self.x_matrices = defaultdict(list)
        else:
            for layer_num in self.x_matrices.keys():
                mask_len = len(self.layer_correlations[layer_num])
                self.masks[layer_num] = self._generate_mask(layer_num, mask_len)

    def unfreeze(self, dry_run):
        self.freezed = False
        if dry_run:
            for layer_num, matrices in self.x_matrices.items():
                x_matrix = np.concatenate(matrices)
                self._setup_dpp(x_matrix, layer_num)

    def _setup_dpp(self, x_matrix, layer_num):
        self.x_matrix = x_matrix
        micro = 1e-12
        x_matrix += np.random.random(x_matrix.shape) * micro  # for computational stability
        if self.covariance:
            L = np.cov(x_matrix.T)
        else:
            L = np.corrcoef(x_matrix.T)

        self.dpps[layer_num] = FiniteDPP('likelihood', **{'L': L})
        self.layer_correlations[layer_num] = L

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.ht_norm:
            L = torch.DoubleTensor(L).to(device)
            I = torch.eye(len(L)).double().to(device)
            K = torch.mm(L, torch.inverse(L + I))

            self.norm[layer_num] = torch.reciprocal(torch.diag(K))  # / len(correlations)
            self.L = L
            self.K = K

    def _generate_mask(self, layer_num, mask_len):
        # sampling nodes ids
        dpp = self.dpps[layer_num]

        for _ in range(ATTEMPTS):
            dpp.sample_exact()
            ids = dpp.list_of_samples[-1]
            if len(ids):  # We should retry if mask is zero-length
                break

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask = torch.zeros(mask_len).double().to(device)
        if self.ht_norm:
            mask[ids] = self.norm[layer_num][ids]
        else:
            mask[ids] = mask_len / len(ids)

        return mask

    def reset(self):
        self.layer_correlations = {}


def get_nu(eigen_values, k):
    """
    Get tilting coefficient to correctly approximate k-dpp marginal probabilities
    See amblard2018

    :param eigen_values: eigen values of L (likelihood) matrix for dpp
    :param k: how much samples do we plan to take
    :return: tilting coefficient
    """
    values = eigen_values + 1e-14
    def point(nu):
        exp_nu = np.exp(nu)
        expect = np.sum([val*exp_nu / (1 + exp_nu*val) for val in values])
        return expect - k

    try:
        solution = root_scalar(point, bracket=[-10., 10.])
        assert solution.converged
        return solution.root
    except (ValueError, AssertionError):
        raise ValueError('k-dpp: Probably too small matrix rank for the k')


class KDPPMask:
    def __init__(self, noise_level=None, tol_level=1e-3, ht_norm=False, covariance=False):
        self.layer_correlations = {}
        self.dry_run = True
        self.dpps = {}
        self.ranks = {}
        self.ranks_history = defaultdict(list)
        self.noise_level = noise_level
        self.tol_level = tol_level

        self.ht_norm = ht_norm
        self.norm = {}

        self.covariance = covariance

        ## For batch loaders
        self.freezed = False
        self.masks = {}
        self.ks = {}

    def freeze(self, dry_run):
        self.freezed = True
        if dry_run:
            self.x_matrices = defaultdict(list)
        else:
            for layer_num in self.x_matrices.keys():
                mask_len = len(self.layer_correlations[layer_num])
                k = self.ks[layer_num]
                self.masks[layer_num] = self._generate_mask(layer_num, mask_len, k)

    def unfreeze(self, dry_run):
        self.freezed = False
        if dry_run:
            for layer_num, matrices in self.x_matrices.items():
                x_matrix = np.concatenate(matrices)
                k = self.ks[layer_num]
                self._setup_dpp(x_matrix, layer_num, k)

    def _rank(self, dpp=None, eigen_values=None):
        if eigen_values is None:
            eigen_values = dpp.L_eig_vals
        rank = np.count_nonzero(eigen_values > self.tol_level)
        return rank

    def _setup_dpp(self, x_matrix, layer_num, k):
        if self.covariance:
            L = np.cov(x_matrix.T)
        else:
            L = np.corrcoef(x_matrix.T)

        if self.noise_level is not None:
            L += self.noise_level * np.eye(len(L))

        if not self.ht_norm:
            self.dpps[layer_num] = FiniteDPP('likelihood', **{'L': L})
            self.dpps[layer_num].sample_exact()  # to trigger eig values generation
        else:
            eigen_values = np.linalg.eigh(L)[0]
            "Get tilted k-dpp, see amblard2018"
            nu = get_nu(eigen_values, k)
            I = torch.eye(len(L))
            L_tilted = np.exp(nu) * torch.DoubleTensor(L)
            K_tilted = torch.mm(L_tilted, torch.inverse(L_tilted + I)).double()
            self.dpps[layer_num] = FiniteDPP('correlation', **{"K": K_tilted.detach().cpu().numpy()})
            self.norm[layer_num] = torch.reciprocal(torch.diag(K_tilted))
            self.L = L_tilted
            self.K = K_tilted

        # Keep data for debugging
        self.layer_correlations[layer_num] = L

    def _generate_mask(self, layer_num, mask_len, k):
        mask = torch.zeros(mask_len).double()
        ids = self.dpps[layer_num].sample_exact_k_dpp(k)

        if self.ht_norm:
            mask[ids] = self.norm[layer_num][ids]
        else:
            mask[ids] = mask_len / len(ids)
        return mask.to(self.device)

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        mask_len = x.shape[-1]
        k = int(mask_len * (1-dropout_rate))

        if layer_num not in self.layer_correlations:
            x_matrix = x.cpu().numpy()

            self.device = x.device
            if self.freezed:
                self.x_matrices[layer_num].append(x_matrix)
            else:
                self._setup_dpp(x_matrix, layer_num, k)
            mask = torch.ones(mask_len).double().to(x.device)
            self.ks[layer_num] = k
            return mask

        if self.freezed:
            mask = self.masks[layer_num]
        else:
            mask = self._generate_mask(layer_num, mask_len, k)

        return mask

    def reset(self):
        self.layer_correlations = {}
