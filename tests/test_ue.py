import pytest

from alpaca.ue import MCDUE
from alpaca.utils import model_builder


def test_sample_mcdue(simple_conv):
    try:
        model, x_batch = simple_conv
        estimator = MCDUE(model, nn_runs=100, acquisition="bald")
        estimations = estimator(x_batch)
    except Exception as e:
        raise pytest.fail("{0}".format(e))


def test_mcdue_masks(simple_conv):
    try:
        model, x_batch = simple_conv
        model = model_builder.build_model(model, dropout_rate=0.0)
        estimator = MCDUE(model, nn_runs=100)
        estimations = estimator(x_batch)
    except Exception as e:
        raise pytest.fail("{0}".format(e))
