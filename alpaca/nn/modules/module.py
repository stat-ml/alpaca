from typing import Optional
import torch.nn as nn


class Module:
    """
    TODO:
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_mode = False

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

    def ue_mode(self):
        self.uncertainty_mode = True
        return self

    def inf_mode(self):
        self.uncertainty_mode = False
        return self

    # FIXME
    __repr__ = __str__
