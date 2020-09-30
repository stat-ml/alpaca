import torch
import torch.nn as nn
import alpaca.nn as ann

__all__ = ["Linear"]


class Linear(nn.Linear):
    """
    Linear layer with the masked noise
    """

    def __init__(self, *args, dropout_rate=0.0, dropout_mask=None, **kwargs):
        if type(args[0]) is ann.Linear or type(args[0]) is nn.Linear:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args, **kwargs)
        self.dropout_rate = dropout_rate
        self.dropout_mask = dropout_mask

    def forward(self, input):
        out = super().forward(input)
        if self.dropout_mask:
            out = out * self.dropout_mask(out, self.dropout_rate)
        else:
            out = torch.nn.functional.dropout(out, p=self.dropout_rate)
        return out

    @classmethod
    def instantiate(cls, module, dropout_rate=0.0, dropout_mask=None):
        instance = cls(module, dropout_rate=dropout_rate, dropout_mask=dropout_mask)
        return instance
