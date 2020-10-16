import abc
from typing import Union
import torch

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


class MaskLayered(BaseMask):
    __doc__ = r"""
    The base class for nn layered masks
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_correlations: dict = dict()
        self.norm: dict = dict()

    @abc.abstractmethod
    def reset(self):
        """
        Resets layers info/stat
        """
        pass

    def reset(self):
        self.layer_correlations = None
        self._init_run = True


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
        dropout_rate: Union[torch.Tensor, float] = 0.5,
        *,
        is_train=True,
    ) -> torch.Tensor:
        mask_len = x.size(-1)
        k = int(mask_len * (1.0 - dropout_rate))

        if self._init_run is True and is_train is False:
            self._init_run = False
            return self._init_layers(x, mask_len, k)

        mask = torch.zeros(mask_len, dtype=x.dtype, device=x.device)
        inds = torch.multinomial(self.layer_correlation, k, replacement=False)

        if self.ht_norm is True:
            mask[inds] = self.norm[inds]
        else:
            mask[inds] = torch.Tensor([1 / (1 - dropout_rate)]).to(dtype=x.dtype)

        return mask.expand_as(x)

    def _init_layers(self, x: torch.Tensor, mask_len: int, k: int) -> torch.Tensor:
        noise = torch.rand(*x.shape, dtype=x.dtype, device=x.device) * self.eps
        corrs = torch.sum(torch.abs(corrcoef((x + noise).transpose(1, 0))), dim=1)
        scores = torch.reciprocal(corrs)

        if self.scaling:
            scores = (
                4.0 * scores / torch.max(scores)
            )  # TODO: remove hard coding or annotate
        self.layer_correlation = torch.nn.functional.softmax(scores)

        if self.ht_norm:
            # Horvitz-Thopson normalization (1 / marginal_prob for each element)
            probabilities = self.layer_correlation
            samples = max(1000, 4 * x.size(-1))  # TODO: why? (explain 1000)
            self.norm = torch.reciprocal(mc_probability(probabilities, k, samples))

        # Initially we should pass identity mask,
        # otherwise we won't get right correlations for all layers
        return torch.ones(mask_len, dtype=x.dtype, device=x.device)


class DecorrelationMaskScaled(MaskLayered):
    """
    TODO:
    """

    _name_collection = {"decorrelating_sc"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling = True


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
        super().__init__()
        self.ht_norm = ht_norm
        self.lambda_ = lambda_
        self.covariance = covariance

    def __call__(
        self,
        x: torch.Tensor,
        dropout_rate: Union[torch.Tensor, float] = 0.5,
        *,
        is_train=True,
    ) -> torch.Tensor:
        mask_len = x.shape[-1]
        k = int(mask_len * (1 - dropout_rate))

        if self._init_run is True and is_train is False:
            self._init_run = False
            return self._init_layers(x, mask_len, k)

        mask = torch.zeros(mask_len, device=x.device)
        ids = torch.multinomial(self.layer_correlations, k, replacement=False)

        if self.ht_norm:
            mask[ids] = self.norm[ids]
        else:
            mask[ids] = 1 / (1 - dropout_rate)

        return mask.expand_as(x)

    def _init_layers(self, x: torch.Tensor, mask_len: int, k: int):
        if self.covariance:
            K = cov(x.transpose(1, 0))
        else:
            K = corrcoef(x.transpose(1, 0))
        identity = torch.eye(K.size(0))
        leverages_matrix = K @ torch.inverse(K + self.lambda_ * identity)
        probabilities = torch.diagonal(leverages_matrix)
        probabilities = probabilities / torch.sum(probabilities)
        self.layer_correlations = probabilities

        if self.ht_norm:
            # Horvitz-Thopson normalization (1 / marginal_prob for each element)
            probabilities = self.layer_correlations
            samples = max(1000, 4 * x.size(-1))  # TODO: why? (explain 1000)
            self.norm = torch.reciprocal(mc_probability(probabilities, k, samples))

        # Initially we should pass identity mask,
        # otherwise we won't get right correlations for all layers
        return torch.ones(mask_len, device=x.device)


class LeverageScoreMaskCov(LeverageScoreMask):
    _name_collection = {"cov_leverages"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ht_norm = True
        self.lambda_ = 1
        self.covariance = True


class DPPMask(MaskLayered):
    def __init__(self, ht_norm: bool = False, covariance: bool = False):
        self.ht_norm = ht_norm
        self.covariance = covariance

    def __call__(
        self,
        x: torch.Tensor,
        dropout_rate: Union[torch.Tensor, float] = 0.5,
        *,
        is_train=True,
    ) -> torch.Tensor:
        if self._init_run is True and is_train is False:
            self._init_run = False
            return self._init_layers(x)

        # sampling nodes ids
        dpp = self.dpps

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

    def _init_layers(self, x: torch.Tensor, eps: float = 1e-12):
        x += torch.rand(x.shape) * eps
        if self.covariance:
            L = cov(x_matrix.transpose(0, 1))
        else:
            L = corrcoef(x_matrix.transpose(0, 1))

        self.dpps = FiniteDPP("likelihood", **{"L": L})
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
