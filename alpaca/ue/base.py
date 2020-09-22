import abc
from typing import Optional, Callable, Union, Tuple
import torch
import torch.nn as nn

from alpaca.ue import masks
from alpaca.ue import acquisitions

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
    dropout_mask: Optional[str]
        The string defining the key for dropout mask logic
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
        net: nn.Module,
        *,
        nn_runs: int = 25,
        dropout_mask: Optional[str] = None,
        acquisition: Optional[Union[str, Callable]] = None,
        keep_runs: bool = False,
    ):
        self.net = net
        self.nn_runs = nn_runs
        if dropout_mask:
            if dropout_mask not in masks.reg_masks:
                # TODO: move this to exceptions
                raise ValueError("The given mask doesn't exist")
            # initialize the mask
            self.dropout_mask = reg_masks.get(dropout_mask)()
        else:
            self.dropout_mask = None
        self.keep_runs = keep_runs

        # evaluate model for the model
        self.net.eval()

        # set acquisition strategy
        if acquisition is None:
            # set default acquisiiton strategy if not given
            # defined as the attribute for each subclass
            self._acquisition = self._default_acquisition
        elif callable(acquisition):
            self._acquisition = acquisition
        else:
            try:
                self._acquisition = acquisitions.acq_reg[acquisition]
            except KeyError:
                # TODO: move this to exceptions list
                raise ValueError("The given acquisition strategy doesn't exist")

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
