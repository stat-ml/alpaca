import abc
from typing import Optional, Union

import torch
import torch.nn as nn
import numpy as np

from alpaca.utils.functions import corrcoef, mc_probability, cov

reg_masks = {}


class BaseMask(metaclass=abc.ABCMeta):
    __doc__ = r"""
    The base class for masks
    """

    def __new__(cls, *args, **kwargs):
        for name in cls._name_collection:
            if name in reg_masks:
                raise ValueError(
                    "The given mask name: \
                            `{}` exists".format(
                        name
                    )
                )
            reg_masks[name] = cls
        instance = super(BaseMask, cls).__new__(cls, *args, **kwargs)
        return instance

    @abc.abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        *,
        dropout_rate: Union[torch.Tensor, float] = 0.5,
        layer_num: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Performs masked inference logic

        Parameters
        ----------
        x : torch.Tensor
            Tensor to be masked
        dropout_rate : float
            Dropout rate of the binary mask
        layer_num : int
            The index number of the layer to perform dropout

        Returns
        -------

        x_ : torch.Tensor
            Masked tensor
        """
        pass

    @abc.abstractmethod
    def dry_run(self, net: nn.Module, X_pool: torch.Tensor, dropout_rate: float):
        """
        Perform the first run of the masking

        Parameters
        ----------
        net : nn.Module
            nn.Module pytorch nn module the dropout applied in
        X_pool : torch.Tensor
            Input tensor
        dropout_rate : float
            Dropout rate of the binary mask

        """
        pass


class MaskLayered(BaseMask):
    __doc__ = r"""
    The base class for nn layered masks
    """

    def __init__(self):
        super().__init__()
        self.layer_correlations: dict = dict()
        self.norm: dict = dict()

    @abc.abstractmethod
    def reset(self):
        """
        Resets layers info/stat
        """
        pass


class BasicBernoulliMask(BaseMask):
    """
    The implementation of Monte Carlo Dropout (MCD) logic

    More about the behaviours of MCD can be found in: https://arxiv.org/pdf/2008.02627.pdf

    Examples
    --------
    >>> estimator = MCDUE(model, nn_runs=100, acquisition='std', dropout_mask="mc_dropout")
    >>> estimations1 = estimator.estimate(x_batch)
    """

    _name_collection = {"mc_dropout"}

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        x: torch.Tensor,
        *,
        dropout_rate: Union[torch.Tensor, float] = 0.5,
        layer_num: Optional[
            int
        ] = None,  # TODO: remove this, we need to think of better OOP arch here
    ) -> torch.Tensor:
        dropout_rate = torch.as_tensor(dropout_rate)
        p = 1.0 - dropout_rate
        if p.lt(0.0) or p.gt(1.0):
            raise ValueError(
                "Dropout probability has to be between 0 and 1, "
                "but got {}".format(p.item)
            )
        return (
            torch.bernoulli(p.expand(x.size(-1)))
            .div_(p)
            .to(dtype=x.dtype, device=x.device)
        )

    def dry_run(self, net: nn.Module, X_pool: torch.Tensor, dropout_rate: float):
        pass


class DecorrelationMask(MaskLayered):
    """
    TODO:
    """

    _name_collection = {"decorrelating"}

    def __init__(
        self,
        *,
        scaling: bool = False,
        ht_norm: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.scaling = scaling  # use adaptive scaling before softmax
        self.ht_norm = ht_norm
        self.eps = eps

    def __call__(
        self,
        x: torch.Tensor,
        *,
        dropout_rate: Union[torch.Tensor, float] = 0.5,
        layer_num: Optional[int] = None,
    ) -> torch.Tensor:
        mask_len = x.size(-1)
        k = int(mask_len * (1.0 - dropout_rate))

        if layer_num not in self.layer_correlations:
            return self._init_layers(x, layer_num, k, mask_len)

        mask = torch.zeros(mask_len, dtype=x.dtype, device=x.device)
        inds = torch.multinomial(
            self.layer_correlations[layer_num], k, replacement=False
        )

        if self.ht_norm is True:
            mask[inds] = self.norm[layer_num][inds]
        else:
            mask[inds] = torch.Tensor([1 / (1 - dropout_rate)]).to(dtype=x.dtype)

        return mask

    def _init_layers(
        self, x: torch.Tensor, layer_num: int, k: int, mask_len: int
    ) -> torch.Tensor:
        noise = torch.rand(*x.shape, dtype=x.dtype, device=x.device) * self.eps
        corrs = torch.sum(torch.abs(corrcoef((x + noise).transpose())), axis=1)
        scores = torch.reciprocal(corrs)

        if self.scaling:
            scores = (
                4.0 * scores / torch.max(scores)
            )  # TODO: remove hard coding or annotate
        self.layer_correlations[layer_num] = torch.softmax(scores)

        if self.ht_norm:
            # Horvitz-Thopson normalization (1 / marginal_prob for each element)
            probabilities = self.layer_correlations[layer_num]
            samples = max(1000, 4 * x.size(-1))  # TODO: why? (explain 1000)
            self.norm[layer_num] = torch.reciprocal(
                mc_probability(probabilities, k, samples)
            )

        # Initially we should pass identity mask,
        # otherwise we won't get right correlations for all layers
        return torch.ones(mask_len, dtype=x.dtype, device=x.device)

    def reset(self):
        self.layer_correlations.clear()

    def dry_run(self, net: nn.Module, X_pool: torch.Tensor, dropout_rate: float):
        net(X_pool, dropout_rate=dropout_rate, dropout_mask=self)


class DecorrelationMaskScaled(MaskLayered):
    """
    TODO:
    """

    _name_collection = {"decorrelating_sc"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling = True


class DecorrelationMaskHT(MaskLayered):
    """
    TODO:
    """

    _name_collection = {"ht_decorrelating"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling = True
        self.ht_norm = True


class LeverageScoreMask(MaskLayered):
    """
    TODO:
    """

    _name_collection = {"leveragescoremask"}

    def __init__(
        self,
        *,
        ht_norm: bool = True,
        lambda_: int = 1,
        covariance: bool = False,
    ):
        self.ht_norm = ht_norm
        self.lambda_ = lambda_
        self.covariance = covariance

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        mask_len = x.shape[-1]
        k = int(mask_len * (1 - dropout_rate))

        if layer_num not in self.layer_correlations:
            return self._init_layers(
                x,
            )

        mask = torch.zeros(mask_len).double().to(x.device)
        ids = np.random.choice(
            len(mask), k, p=self.layer_correlations[layer_num], replace=False
        )

        if self.ht_norm:
            mask[ids] = torch.DoubleTensor(self.norm[layer_num][ids]).to(x.device)
        else:
            mask[ids] = 1 / (1 - dropout_rate)

        return mask

    def _init_layers(self, x: torch.Tensor, layer_num: int, mask_len: int, k: int):
        if self.covariance:
            K = cov(x.transpose())
        else:
            K = corrcoef(x.transpose())
        identity = torch.eye(K.size(0))
        leverages_matrix = torch.dot(K, torch.inverse(K + self.lambda_ * identity))
        probabilities = torch.diagonal(leverages_matrix)
        probabilities = probabilities / torch.sum(probabilities)
        self.layer_correlations[layer_num] = probabilities

        if self.ht_norm:
            # Horvitz-Thopson normalization (1 / marginal_prob for each element)
            probabilities = self.layer_correlations[layer_num]
            samples = max(1000, 4 * x.size(-1))  # TODO: why? (explain 1000)
            self.norm[layer_num] = torch.reciprocal(
                mc_probability(probabilities, k, samples)
            )

        # Initially we should pass identity mask,
        # otherwise we won't get right correlations for all layers
        return torch.ones(mask_len, device=x.device)

    def reset(self):
        self.layer_correlations = {}

    def dry_run(self, net: nn.Module, X_pool: torch.Tensor, dropout_rate: float):
        net(X_pool, dropout_rate=dropout_rate, dropout_mask=self)


class LeverageScoreMaskHT(LeverageScoreMask):
    _name_collection = {"ht_leverages"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ht_norm = True
        self.lambda_ = 1


class LeverageScoreMaskCov(LeverageScoreMask):
    _name_collection = {"cov_leverages"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ht_norm = True
        self.lambda_ = 1
        self.covariance = True


"""
class DPPMask:
    def __init__(self, ht_norm=False, covariance=False):
        self.dpps = {}
        self.layer_correlations = (
            {}
        )  # keep for debug purposes # Flag for uncertainty estimator to make first run without taking the result
        self.dry_run = True
        self.ht_norm = ht_norm
        self.norm = {}
        self.covariance = covariance

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        if layer_num not in self.layer_correlations:
            # warm-up, generatign correlations masks
            x_matrix = x.cpu().numpy()

            self.x_matrix = x_matrix
            micro = 1e-12
            x_matrix += (
                np.random.random(x_matrix.shape) * micro
            )  # for computational stability
            if self.covariance:
                L = np.cov(x_matrix.T)
            else:
                L = np.corrcoef(x_matrix.T)

            self.dpps[layer_num] = FiniteDPP("likelihood", **{"L": L})
            self.layer_correlations[layer_num] = L

            if self.ht_norm:
                L = torch.DoubleTensor(L).cuda()
                I = torch.eye(len(L)).double().cuda()
                K = torch.mm(L, torch.inverse(L + I))

                self.norm[layer_num] = torch.reciprocal(
                    torch.diag(K)
                )  # / len(correlations)
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

        return x.data.new(mask)

    def reset(self):
        self.layer_correlations = {}


def get_nu(eigen_values, k):
    values = eigen_values + 1e-14

    def point(nu):
        exp_nu = np.exp(nu)
        expect = np.sum([val * exp_nu / (1 + exp_nu * val) for val in values])
        return expect - k

    try:
        solution = root_scalar(point, bracket=[-10.0, 10.0])
        assert solution.converged
        return solution.root
    except (ValueError, AssertionError):
        raise ValueError("k-dpp: Probably too small matrix rank for the k")


class KDPPMask:
    def __init__(
        self, noise_level=None, tol_level=1e-3, ht_norm=False, covariance=False
    ):
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
        k = int(mask_len * (1 - dropout_rate))

        if layer_num not in self.layer_correlations:
            x_matrix = x.cpu().numpy()

            if self.covariance:
                L = np.cov(x_matrix.T)
            else:
                L = np.corrcoef(x_matrix.T)

            if self.noise_level is not None:
                L += self.noise_level * np.eye(len(L))

            if not self.ht_norm:
                self.dpps[layer_num] = FiniteDPP("likelihood", **{"L": L})
                self.dpps[layer_num].sample_exact()  # to trigger eig values generation
            else:
                eigen_values = np.linalg.eigh(L)[0]
                "Get tilted k-dpp, see amblard2018"
                nu = get_nu(eigen_values, k)
                I = torch.eye(len(L)).to(x.device)
                L_tilted = np.exp(nu) * torch.DoubleTensor(L).to(x.device)
                K_tilted = torch.mm(L_tilted, torch.inverse(L_tilted + I)).double()
                self.dpps[layer_num] = FiniteDPP(
                    "correlation", **{"K": K_tilted.detach().cpu().numpy()}
                )
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
"""
