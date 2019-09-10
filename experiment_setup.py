import random

import numpy as np
import torch

from model.mlp import MLP
from uncertainty_estimator.nngp import NNGPRegression
from uncertainty_estimator.mcdue import MCDUE, MCDUEMasked
from uncertainty_estimator.random_estimator import RandomEstimator


def build_estimator(name, model, **kwargs):
    if name == 'nngp':
        estimator = NNGPRegression(model, **kwargs)
    elif name == 'random':
        estimator = RandomEstimator()
    elif name == 'mcdue':
        estimator = MCDUE(model, **kwargs)
    elif name == 'mcdue_masked':
        estimator = MCDUEMasked(model, **kwargs)
    else:
        raise ValueError("Wrong estimator name")
    return estimator


def set_random(random_seed):
    # Setting seeds for reproducibility
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)


def get_model(layers, model_path, train_set, val_set, retrain=False, l2_reg=1e-5, **kwargs):
    model = MLP(layers, l2_reg=l2_reg)
    if retrain:
        model.fit(train_set, val_set, **kwargs)
        torch.save(model.state_dict(), model_path)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
    return model

