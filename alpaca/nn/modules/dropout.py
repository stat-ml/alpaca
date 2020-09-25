import torch
import torch.nn.functional as F

__all__ = ["Dropout"]


class Dropout:
    """
    The implementation of dropout with an additional `dropout_mask` parameterization
    """

    def __init__(self, dropout_rate, layer_indx=None, dropout_mask=None):
        self.dropout_rate = dropout_rate
        self.dropout_mask = dropout_mask
        self.layer_indx = layer_indx

    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.dropout_mask and self.layer_indx is None:
            raise ValueError("You need to set `layer_num` for masked dropout")
        if self.dropout_mask is None:
            return F.dropout(x, self.dropout_rate)
        else:
            return self.dropout_mask(
                x, dropout_rate=self.dropout_rate, layer_num=self.layer_indx
            )
