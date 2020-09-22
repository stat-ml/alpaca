from typing import Optional
import torch
import torch.nn.functional as F
from alpaca.ue.masks import BaseMask

__all__ = ["Dropout"]


class Dropout:
    """
        The implementation of dropout with an additional `dropout_mask` parameterization
    """
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def __call__(
        self,
        x: torch.Tensor,
        *,
        dropout_mask: Optional[BaseMask] = None,
        layer_num: Optional[int] = None,
    ) -> torch.Tensor:
        if dropout_mask and layer_num is None:
            raise ValueError("You need to set `layer_num` for masked dropout")
        if dropout_mask is None:
            return F.dropout(x, self.dropout_rate)
        else:
            return dropout_mask(x, self.dropout_rate, layer_num)
