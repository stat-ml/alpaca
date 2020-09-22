import torch
from alpaca.nn import Dropout


def test_dropout_extrs():
    module = Dropout(0.)
    input = torch.randn(10)
    res = module(input)
    assert res.sum() == input.sum()

    module = Dropout(1.)
    res = module(input)
    assert res.sum().item() == 0.
