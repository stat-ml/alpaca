from .nngp import NNGPRegression
from .mcdue import MCDUE, MCDUEMasked
from .eue import EnsembleUE, EnsembleMCDUE, EnsembleNLLUE, EnsembleMCDUEMasked
from .random_estimator import RandomEstimator
from .bald import Bald, BaldMasked, BaldEnsemble


def build_estimator(name, model, **kwargs):
    if name == 'nngp':
        estimator = NNGPRegression(model, **kwargs)
    elif name == 'random':
        estimator = RandomEstimator()
    elif name == 'mcdue':
        estimator = MCDUE(model, **kwargs)
    elif name == 'mcdue_masked':
        estimator = MCDUEMasked(model, **kwargs)
    elif name == 'bald':
        estimator = Bald(model, **kwargs)
    elif name == 'bald_masked':
        estimator = BaldMasked(model, **kwargs)
    elif name == 'bald_ensemble':
        estimator = BaldEnsemble(model, **kwargs)
    elif name == 'eue_nll':
        estimator = EnsembleNLLUE(model)
    elif name == 'eue':
        estimator = EnsembleUE(model)
    elif name == 'emcdue':
        estimator = EnsembleMCDUE(model, **kwargs)
    elif name == 'emcdue_masked':
        estimator = EnsembleMCDUEMasked(model, **kwargs)
    else:
        raise ValueError("Wrong estimator name")
    return estimator
