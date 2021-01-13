from collections import defaultdict
import abc
from typing import Union
import torch
import numpy as np
from scipy.optimize import root_scalar

from alpaca.utils.functions import corrcoef, mc_probability, cov
from dppy.finite_dpps import FiniteDPP

__all__ = ["reg_masks"]

reg_masks = {}


def register_mask(cls):
    for key in cls._name_collection:
        reg_masks[key] = cls.__name__


class BaseMask(metaclass=abc.ABCMeta):
    __doc__ = r"""
    The base class for masks
    """

    def __init__(self):
        self._init_run = True

    @abc.abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        *,
        dropout_rate: Union[torch.Tensor, float] = 0.5,
    ) -> torch.Tensor:
        """
        Performs masked inference logic

        Parameters
        ----------
        x : torch.Tensor
            Tensor to be masked
        dropout_rate : float
            Dropout rate of the binary mask

        Returns
        -------

        x_ : torch.Tensor
            Masked tensor
        """
        pass

    def copy(self) -> "BaseMask":
        """
        Creates the copy of an instance
        """
        instance = self.__class__()
        instance.__dict__ = self.__dict__.copy()
        return instance

    def reset(self):
        pass


class DecorrelationMask(BaseMask):
    pass

class LeverageScoreMask(BaseMask):
    pass

class BasicBernoulliMask(BaseMask):
    """
    The implementation of Monte Carlo Dropout (MCD) logic

    More about the behaviours of MCD can be found in: https://arxiv.org/pdf/2008.02627.pdf

    Examples
    --------
    >>> estimator = MCDUE(model, nn_runs=100, acquisition='std')
    >>> estimations1 = estimator.estimate(x_batch)
    """

    _name_collection = {"mc_dropout"}

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        x: torch.Tensor,
        dropout_rate: Union[torch.Tensor, float] = 0.5,
        *,
        is_train=True,
    ) -> torch.Tensor:
        dropout_rate = torch.as_tensor(dropout_rate)
        p = 1.0 - dropout_rate
        if p.lt(0.0) or p.gt(1.0):
            raise ValueError(
                "Dropout probability has to be between 0 and 1, "
                "but got {}".format(p.item)
            )
        res = (
            torch.bernoulli(p.expand(x.shape))
            .div_(p)
            .to(dtype=x.dtype, device=x.device)
        )
        return res


ATTEMPTS = 5
class DPPMask:
    def __init__(self, ht_norm=True, covariance=False):
        self.dpps = {}
        self.layer_correlations = {}  # keep for debug purposes # Flag for uncertainty estimator to make first run without taking the result
        self.dry_run = True
        self.ht_norm = ht_norm
        self.norm = {}
        self.covariance = covariance
        self._init_run = True

    def __call__(self, x, dropout_rate=0.5, layer_num=0, is_train=True):
        if layer_num not in self.layer_correlations:
            # warm-up, generatign correlations masks
            x_matrix = x.cpu().numpy()

            self.x_matrix = x_matrix
            micro = 1e-12
            x_matrix += np.random.random(x_matrix.shape) * micro  # for computational stability
            if self.covariance:
                L = np.cov(x_matrix.T)
            else:
                L = np.corrcoef(x_matrix.T)

            self.dpps[layer_num] = FiniteDPP('likelihood', **{'L': L})
            self.layer_correlations[layer_num] = L

            if self.ht_norm:
                L = torch.DoubleTensor(L).cuda()
                I = torch.eye(len(L)).double().cuda()
                K = torch.mm(L, torch.inverse(L + I))

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

        mask_len = x.shape[-1]
        mask = torch.zeros(mask_len).double().cuda()
        if self.ht_norm:
            mask[ids] = self.norm[layer_num][ids]
        else:
            mask[ids] = mask_len / len(ids)

        # return x.data.new(mask)
        return mask.to(x.device)

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
    def __init__(self, noise_level=None, tol_level=1e-3, ht_norm=True, covariance=False):
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

    def _rank(self, dpp=None, eigen_values=None):
        if eigen_values is None:
            eigen_values = dpp.L_eig_vals
        rank = np.count_nonzero(eigen_values > self.tol_level)
        return rank

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        mask_len = x.shape[-1]
        k = int(mask_len * (1-dropout_rate))

        if layer_num not in self.layer_correlations:
            x_matrix = x.cpu().numpy()

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
                I = torch.eye(len(L)).to(x.device)
                L_tilted = np.exp(nu) * torch.DoubleTensor(L).to(x.device)
                K_tilted = torch.mm(L_tilted, torch.inverse(L_tilted + I)).double()
                self.dpps[layer_num] = FiniteDPP('correlation', **{"K": K_tilted.detach().cpu().numpy()})
                self.norm[layer_num] = torch.reciprocal(torch.diag(K_tilted))
                self.L = L_tilted
                self.K = K_tilted

            # Keep data for debugging
            self.layer_correlations[layer_num] = L
            mask = torch.ones(mask_len).double().to(x.device)

            return mask

        mask = torch.zeros(mask_len).double().to(x.device)
        ids = self.dpps[layer_num].sample_exact_k_dpp(k)

        if self.ht_norm:
            mask[ids] = self.norm[layer_num][ids]
        else:
            mask[ids] = mask_len / len(ids)

        return mask

    def reset(self):
        self.layer_correlations = {}
