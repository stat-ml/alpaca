from typing import Optional
import torch
import torch.nn.functional as F
from alpaca.ue.masks import BaseMask

__all__ = ["Dropout"]


class Dropout:
    """
    The implementation of dropout with an additional `dropout_mask` parameterization
    """

    def __init__(
        self, dropout_rate: float = 0.0, dropout_mask: Optional[BaseMask] = None
    ):
        self.dropout_rate = dropout_rate
        self.dropout_mask = dropout_mask

    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.dropout_mask is None:
            return F.dropout(x, self.dropout_rate)
        else:
            return self.dropout_mask(x, dropout_rate=self.dropout_rate)
