from typing import Optional
import torch.nn as nn


class Module:
    """
    TODO:
    """

    def instantiate_with_dropout_params(
        self,
        module: nn.Module,
        dropout_rate: float = 0.0,
        dropout_mask: "BaseMask" = None,
    ):
        self.__dict__ = module.__dict__.copy()
        self.dropout_rate = dropout_rate
        self.dropout_mask = dropout_mask
        return self

    def __str__(self):
        return "ann.{}, dropout_rate: {}, dropout_mask: {}".format(
            self.__class__.__name__,
            self.dropout_rate,
            self.dropout_mask.__class__.__name__,
        )

    # FIXME
    __repr__ = __str__
