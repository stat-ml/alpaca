import pytest
from alpaca.ue import MCDUE

def test_sample_mcdue(simple_conv):
    try:
        model, x_batch = simple_conv
        estimator = MCDUE(model, nn_runs=100, acquisition="bald")
        estimations = estimator.estimate(x_batch)
    except Exception as e:
        raise pytest.fail("{0}".format(e))
