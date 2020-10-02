from typing import Optional, List
import torch.nn as nn

from alpaca.ue.masks import BaseMask
from alpaca.nn.modules.module import Module


def build_model(
    model: nn.Module,
    dropout_rate: float = 0.0,
    dropout_mask: Optional[BaseMask] = None,
    *,
    keys: Optional[List[str]] = None,
):
    """
    Replaces all the ann.Modules in the model with the copy and
    the new dropout paremetrization

    Parameters
    ----------
    model: nn.Module
        The model which modules should be parametrized
    dropout_rate: float = 0.0
        The dropout rate parameterization
    dropout_mask: Optional[BaseMask] None
        The dropout_mask parameterization
    keys: Optional[List[str]] = None
        The keys of the modules in the model which should be parametrized
    """
    for key, item in model._modules.items():
        if isinstance(item, Module):
            if keys and key not in keys:
                continue
            model._modules[key] = item.instantiate_with_dropout_params(
                item,
                dropout_rate=dropout_rate,
                dropout_mask=dropout_mask.copy() if dropout_mask else None,
            )
    return model