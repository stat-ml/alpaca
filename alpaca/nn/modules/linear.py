from typing import Optional
import torch
import torch.nn as nn

from alpaca.nn.modules.module import Module

__all__ = ["Linear"]


class Linear(nn.Linear, Module):
    """
    The subclass of nn.Linear layer with the additional `dropout_mask` and `dropout_rate` parameterization
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

    def forward(self, input: torch.Tensor):
        out = super().forward(input)
        if self.dropout_mask is None or self.training is True:
            out = torch.nn.functional.dropout(out, p=self.dropout_rate)
        else:
            out = out * self.dropout_mask(
                out, self.dropout_rate, is_train=self.training
            )
        return out
