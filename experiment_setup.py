import os
import random

import numpy as np
import torch

from model.mlp import MLP
from uncertainty_estimator.nngp import NNGPRegression
from uncertainty_estimator.mcdue import MCDUE, MCDUEMasked
from uncertainty_estimator.eue import EnsembleUE, EnsembleMCDUE
from uncertainty_estimator.random_estimator import RandomEstimator

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_estimator(name, model, **kwargs):
    if name == 'nngp':
        estimator = NNGPRegression(model, **kwargs)
    elif name == 'random':
        estimator = RandomEstimator()
    elif name == 'mcdue':
        estimator = MCDUE(model, **kwargs)
    elif name == 'mcdue_masked':
        estimator = MCDUEMasked(model, **kwargs)
    elif name == 'eue':
        estimator = EnsembleUE(model, **kwargs)
    elif name == 'emcdue':
        estimator = EnsembleMCDUE(model, **kwargs)
    else:
        raise ValueError("Wrong estimator name")
    return estimator


def set_random(random_seed):
    # Setting seeds for reproducibility
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)


def get_model(model, model_path, train_set=None, val_set=None, retrain=False, **kwargs):
    model_path = os.path.join(ROOT_DIR, model_path)
    if retrain:
        if train_set is None or val_set is None:
            raise RuntimeError("You should pass datasets for retrain")
        model.fit(train_set, val_set, **kwargs)
        torch.save(model.state_dict(), model_path)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
    return model

