from typing import Optional
import torch
import torch.nn as nn

from alpaca.nn.modules.module import Module

__all__ = ["Dropout"]


class Dropout(Module, nn.Dropout):
    """
    The subclass of nn.Dropout layer with the additional `dropout_mask` and `dropout_rate` parameterization
    """

    def __init__(
        self,
        *args,
        dropout_rate: float = 0.0,
        dropout_mask: "BaseMask" = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dropout_rate = dropout_rate
        self.dropout_mask = dropout_mask

    def __call__(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        if self.training:
            return torch.nn.functional.dropout(
                input, p=self.dropout_rate, inplace=self.inplace
            )
        else:
            if self.uncertainty_mode is True and self.dropout_mask:
                return input * self.dropout_mask(
                    input, dropout_rate=self.dropout_rate, is_train=self.training
                )
            else:
                return input
