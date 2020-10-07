import pytest
import numpy as np
import torch
import torch.nn as nn

from alpaca.utils import model_builder
import alpaca.nn as ann


@pytest.fixture(scope="module")
def simple_model():
    net = nn.Sequential(nn.Linear(10, 10), ann.Dropout())
    return net


@pytest.fixture(scope="module")
def nn_simple_model():
    def _nn_simple_model(dropout_rate):
        net = nn.Sequential(nn.Linear(10, 10), nn.Dropout(p=dropout_rate))
        return net

    return _nn_simple_model


@pytest.fixture(scope="module")
def ann_dropout():
    net = nn.Sequential(ann.Dropout())
    return net


@pytest.fixture(scope="module")
def nn_dropout():
    def _nn_net(dropout_rate):
        net = nn.Sequential(nn.Dropout(p=dropout_rate))
        return net

    return _nn_net


@torch.no_grad()
def test_check_model_inference(simple_model):
    x = torch.randn((1, 10))
    out = simple_model(x)
    assert out.shape == (1, 10)


@torch.no_grad()
def test_check_build_zero_p(simple_model, seed):
    simple_model = model_builder.build_model(simple_model, dropout_rate=1.0)
    x = torch.randn((1, 10))
    out = simple_model(x)
    np.testing.assert_allclose(out.sum(), 0.0, rtol=1e-6)


@torch.no_grad()
def test_check_output_with_nn(ann_dropout, nn_dropout, seed, dropout_rate_extreme):
    dropout_rate = dropout_rate_extreme
    x = torch.randn((1, 10))
    nn_dropout = nn_dropout(dropout_rate)
    ann_dropout = model_builder.build_model(ann_dropout, dropout_rate=dropout_rate)
    out1, out2 = ann_dropout(x), nn_dropout(x)
    np.testing.assert_allclose(out1, out2, rtol=1e-6)
