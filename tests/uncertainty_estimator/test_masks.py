import torch
from alpaca.uncertainty_estimator.masks import BasicMask


def test_basic_mask():
    mask = BasicMask()
    x = torch.Tensor(range(6))
    result = mask(x, dropout_rate=0.33)

    assert result.shape == (6,)
    assert sum(result) == 6
    assert len(result[result != 0]) == 4
