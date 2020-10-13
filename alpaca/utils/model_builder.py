from typing import Optional, List
import torch.nn as nn

from alpaca.ue.masks import BaseMask
from alpaca.nn.modules.module import Module
from alpaca.models import Ensemble


def build_model(
    model,
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

    if isinstance(model, Ensemble):
        for i, _ in enumerate(model.modules):
            model.modules[i] = build_model(
                _, dropout_rate=dropout_rate, dropout_mask=dropout_mask, keys=keys
            )
        return model

    for key, item in model._modules.items():
        if isinstance(item, Module):
            if keys and key not in keys:
                continue
            model._modules[key] = item.instantiate_with_dropout_params(
                item,
                dropout_rate=dropout_rate,
                dropout_mask=dropout_mask.copy() if dropout_mask else None,
            )
        elif type(item) == nn.Sequential or type(item) == nn.ModuleList:
            for i, module in enumerate(item):
                module = build_model(
                    module,
                    dropout_rate=dropout_rate,
                    dropout_mask=dropout_mask,
                    keys=keys,
                )
                model._modules[key][i] = module
        else:
            pass
    return model


def uncertainty_mode(model: nn.Module):
    if isinstance(model, Ensemble):
        for i, _ in enumerate(model.modules):
            model.modules[i] = uncertainty_mode(_)
        return model

    for key, item in model._modules.items():
        if isinstance(item, Module):
            model._modules[key] = item.ue_mode()
        elif type(item) == nn.Sequential or type(item) == nn.ModuleList:
            for i, module in enumerate(item):
                module = uncertainty_mode(module)
                model._modules[key][i] = module
        else:
            pass
    return model


def inference_mode(model: nn.Module):
    if isinstance(model, Ensemble):
        for i, _ in enumerate(model.modules):
            model.modules[i] = inference_mode(_)
        return model

    for key, item in model._modules.items():
        if isinstance(item, Module):
            model._modules[key] = item.inf_mode()
        elif type(item) == nn.Sequential or type(item) == nn.ModuleList:
            for i, module in enumerate(item):
                module = inference_mode(module)
                model._modules[key][i] = module
        else:
            pass
    return model
