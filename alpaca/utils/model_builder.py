from typing import Optional, List, Union
import torch.nn as nn

from alpaca.ue.masks import BaseMask
from alpaca.nn.modules.module import Module
from alpaca.models import EnsembleConstructor


def build_model(
    model,
    dropout_rate: Union[List[float], float] = 0.0,
    dropout_mask: Optional[Union[List[BaseMask], BaseMask]] = None,
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
    dropout_rate: Any[List[float], float] = 0.0
        The dropout rate parameterization
    dropout_mask: Optional[Any[List[BaseMask], BaseMask]] = None
        The dropout_mask parameterization
    keys: Optional[List[str]] = None
        The keys of the modules in the model which should be parametrized
    """

    if isinstance(model, EnsembleConstructor):
        for i, _ in enumerate(model.modules):
            model.modules[i] = build_model(
                _, dropout_rate=dropout_rate, dropout_mask=dropout_mask, keys=keys
            )
        return model

    index_dropout = 0

    for key, item in model._modules.items():
        if isinstance(item, Module):
            if keys and key not in keys:
                continue
            if isinstance(dropout_rate, list) and index_dropout >= len(dropout_rate):
                raise ValueError(
                    "Model contains more stochastic layers than it is provided"
                )
            if dropout_mask:
                mask = (
                    dropout_mask
                    if not isinstance(dropout_mask, list)
                    else dropout_mask[index_dropout]
                )
            else:
                mask = None
            model._modules[key] = item.instantiate_with_dropout_params(
                item,
                dropout_rate=dropout_rate
                if not isinstance(dropout_rate, list)
                else dropout_rate[index_dropout],
                dropout_mask=mask,
            )
            index_dropout += 1
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
    if isinstance(model, EnsembleConstructor):
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
    if isinstance(model, EnsembleConstructor):
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
