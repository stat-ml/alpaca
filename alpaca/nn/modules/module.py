from typing import Optional
import torch.nn as nn

from alpaca.ue.masks import BaseMask


class Module:
    """
    TODO:
    """

    def instantiate_with_dropout_params(
        self,
        module: nn.Module,
        dropout_rate: float = 0.0,
        dropout_mask: Optional[BaseMask] = None,
    ):
        self.__dict__ = module.__dict__.copy()
        self.dropout_rate = dropout_rate
        self.dropout_mask = dropout_mask
        return self

    @property
    def _extra_repr():
        return ""

    def __repr__(self):
        extra = self._extra_repr
        if extra:
            return "{} layer; {}".format(self.__class__.__name__, extra)
        else:
            return "{} layer".format(self.__class__.__name__)
