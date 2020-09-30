from typing import Optional
import torch.nn as nn
from alpaca.ue.masks import BaseMask
import alpaca.nn as ann


def build_model(
    model: nn.Module, dropout_mask: Optional[BaseMask] = None, dropout_rate: float = 0.0
):
    for key, item in model._modules.items():
        if type(item) is ann.Linear or type(item) is ann.Dropout:
            model._modules[key] = item.__class__.instantiate(
                item, dropout_rate=dropout_rate, dropout_mask=dropout_mask
            )
    return model
