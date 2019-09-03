import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from model.mlp import MLP
from dataloader.rosen import RosenData
from uncertainty_estimator.nngp import NNGPRegression
from uncertainty_estimator.mcdue import MCDUE, MCDUEMasked
from uncertainty_estimator.random_estimator import RandomEstimator
from analysis.autoencoder import VAE

config = {
    'estimator': 'nngp',
    'random_seed': 43,
    'n_dim': 10,
    'data_size': 2000,
    'data_split': [0.4, 0.1, 0.1, 0.4],
    'update_size': 100,
    'al_iterations': 10,
    'verbose': True,
    'use_cache': True,
    'layers': [10, 128, 64, 32, 1],
    'patience': 5,
    'retrain': True,
    'model_path': 'model/data/rosen_visual.ckpt'
}


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


def get_model(layers, model_path, train_set, val_set, epochs=10000, retrain=False):
    model = MLP(layers)
    if retrain:
        model.fit(train_set, val_set, epochs=epochs)
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    return model


def get_vae(restore, rosen, patience, epochs=100):
    x_pool, y_pool = rosen.dataset('pool')
    x_train, y_train = rosen.dataset('train')
    x_val, y_val = rosen.dataset('train')
    x = np.concatenate((x_train, x_pool))

    vae_path = 'model/data/vae.ckpt'
    vae = VAE(10, 10, 2)
    if restore:
        vae.load_state_dict(torch.load(vae_path))
    else:
        current_patience = patience
        best_loss = float('inf')

        for epoch in range(epochs):
            vae.fit(x)
            if (epoch+1) % 100 == 0:
                val_loss = vae.evaluate(x_val)
                print('{} ====> Val set loss: {:.4f}'.format(epoch+1, val_loss))
                if val_loss < best_loss:
                    best_loss = val_loss
                    current_patience = patience
                else:
                    current_patience -= 1
                    if current_patience <= 0:
                        print("No patience left")
                        break
        torch.save(vae.state_dict(), vae_path)
    return vae


if __name__ == '__main__':
    epochs = 100_000
    patience = 5
    rosen = RosenData(
        config['n_dim'], config['data_size'], config['data_split'],
        use_cache=config['use_cache'])
    x_pool, y_pool = rosen.dataset('pool')
    x_train, y_train = rosen.dataset('train')
    x_val, y_val = rosen.dataset('train')

    set_random(config['random_seed'])
    x_full = np.concatenate((x_train, x_pool))
    y_full = np.concatenate((y_train, y_pool))
    model = get_model(config['retrain'], config['model_path'], (x_full, y_full), (x_val, y_val))
    estimator = build_estimator(config['estimator'], model)
    estimation = estimator.estimate(x_pool, x_train, y_train)

    points = 5
    print(model(x_pool[:points]), y_pool[:points])

