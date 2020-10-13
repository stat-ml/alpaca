import abc
from collections.abc import Iterable
from typing import Tuple
import torch
import torch.nn as nn

from alpaca.nn.modules.module import Module


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

        self._masks_collect()
        self.reset()

    @abc.abstractmethod
    def __call__(self, X_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _masks_collect_helper(self, model):
        if not isinstance(model, Iterable):
            model = [model]

        for model_ in model:
            for key, item in model_._modules.items():
                if isinstance(item, Module):
                    self.all_masks.add(item.dropout_mask)
                elif type(item) == nn.Sequential or type(item) == nn.ModuleList:
                    for i, module in enumerate(item):
                        self._masks_collect_helper(module)

    def _masks_collect(self):
        self.all_masks = set()
        self._masks_collect_helper(self.net)

    def reset(self):
        """
        Resets stats of all masks in the model
        """
        for item in self.all_masks:
            if item:
                item.reset()
