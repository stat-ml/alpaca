"""
    TODO: Docs
"""
import torch

__all__ = [
    "var_ratio",
    "var_soft",
    "bald",
    "bald_normed",
]

# store all acquisitions in dict
acq_reg = {}


def reg_acquisition(f):
    acq_reg[f.__name__] = f
    return f


@reg_acquisition
def std(mcd_runs: torch.Tensor):
    """
    var_ratio_acquisition
    TODO: docs
    """
    return mcd_runs.std(dim=0)


@reg_acquisition
def var_ratio(mcd_runs: torch.Tensor, nn_runs: int):
    predictions = torch.argmax(mcd_runs, axis=-1)
    # count how many time repeats the strongest class
    mode_count = lambda preds: torch.max(torch.bincount(preds))
    modes = [mode_count(point) for point in predictions]
    ue = 1 - torch.stack(modes) / nn_runs
    return ue


@reg_acquisition
def var_soft(mcd_runs: torch.Tensor):
    """
    var_soft_acquisition
    TODO: docs
    """
    probabilities = torch.softmax(mcd_runs, axis=-1)
    ue = torch.mean(torch.std(probabilities, dim=-2), dim=-1)
    return ue


@reg_acquisition
def bald(mcd_runs: torch.Tensor):
    """
    bald_acquisition
    TODO: docs
    """
    return _bald(mcd_runs)


@reg_acquisition
def bald_normed(mcd_runs: torch.Tensor):
    """
    bald_normed_acquisition
    TODO: docs
    """
    return _bald_normed(mcd_runs)


def _entropy(x):
    return torch.sum(-x * torch.log(torch.clamp(x, 1e-8, 1)), dim=-1)


def _bald(logits):
    predictions = torch.softmax(logits, dim=-1)
    predictive_entropy = _entropy(torch.mean(predictions, dim=1))
    expected_entropy = torch.mean(_entropy(predictions), dim=1)
    return predictive_entropy - expected_entropy


def _bald_normed(logits):
    predictions = torch.softmax(logits, dim=-1)

    predictive_entropy = _entropy(torch.mean(predictions, dim=1))
    expected_entropy = torch.mean(_entropy(predictions), dim=1)

    return (predictive_entropy - expected_entropy) / predictive_entropy
