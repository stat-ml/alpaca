import torch
from alpaca.uncertainty_estimator.masks import BasicBernoulliMask


def test_basic_mask():
    mask = BasicBernoulliMask()
    x = torch.Tensor(range(6))
    result = mask(x, dropout_rate=0.33)
    assert result.shape == (6,)
