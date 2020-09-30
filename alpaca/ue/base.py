import abc
from typing import Tuple
import torch
import torch.nn as nn


__all__ = ["UE"]


class UE(metaclass=abc.ABCMeta):
    """
    Abstract class for all uncertainty estimation method implementations

    Parameters
    ----------
    net: :class:`torch.nn.Module`
        Neural network on based on which we are calculating uncertainty region
    nn_runs: int
        A number of iterations
    acquisitions: Optional[Union[str, Callable]]
        Acquisiiton function definition
    keep_runs: bool
        Whenever to save iteration results

    Default attributes
    ------------------
    _name : None
    _default_acquisition : None
    """

    _name = None
    _default_acquisition = None

    def __init__(
        self,
        net,
        *,
        nn_runs: int = 25,
        keep_runs: bool = False,
    ):
        # we are keeping list for ensemble estimators
        self.net = net
        self.nn_runs = nn_runs
        self.keep_runs = keep_runs

        if isinstance(net, nn.Module):
            # evaluate model for the model
            self.net.eval()

    @abc.abstractmethod
    def estimate(self, X_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty

        Parameters
        ----------
        X_pool: torch.Tensor
            Batch of tensor based on which the uncertainty is estimated

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
        """
        pass

    def reset(self):
        """
        Reset dropout mask stats
        """
        if self.dropout_mask:
            self.dropout_mask.reset()

    def last_mcd_runs(self):
        """
        Return model prediction for last uncertainty estimation
        """
        if not self.keep_runs:
            raise ValueError(
                "mcd_runs: You should set `keep_runs=True` to properly use this method"
            )
        return self._mcd_runs

    @property
    def mcd_runs(self):
        if hasattr(self, "_mcd_runs"):
            # TODO we an add logger here to inform a user
            # that the `keep_runs` flag is False
            return self._mcd_runs
        return None

    @property
    def desc(self):
        """
        Description of the UE algorithm
        """
        return "Uncertainty estimation with {} approach".format(self._name)
