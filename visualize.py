import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from model.mlp import MLP
from dataloader.rosen import RosenData
from uncertainty_estimator.nngp import NNGPRegression
from uncertainty_estimator.mcdue import MCDUE
from uncertainty_estimator.random_estimator import RandomEstimator
from analysis.autoencoder import VAE

config = {
    'estimator': 'nngp',
    'random_seed': 43,
    'n_dim': 10,
    'data_size': 2000,
    'data_split': [0.2, 0.1, 0.1, 0.6],
    'update_size': 100,
    'al_iterations': 10,
    'verbose': True,
    'use_cache': True,
    'layers': [10, 128, 64, 32, 1],
    'patience': 5,
    'retrain': False,
    'model_path': 'model/data/rosen_visual.ckpt'
}


def build_estimator(name, model):
    if name == 'nngp':
        estimator = NNGPRegression(model)
    elif name == 'random':
        estimator = RandomEstimator()
    elif name == 'mcdue':
        estimator = MCDUE(model)
    else:
        raise ValueError("Wrong estimator name")
    return estimator


def set_random(random_seed):
    # Setting seeds for reproducibility
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)


def get_model(retrain, model_path, train_set, val_set):
    model = MLP(config['layers'])
    if retrain:
        model.fit(train_set, val_set)
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    return model


def get_vae(restore, rosen, patience):
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
    # rosen = RosenData(
    #     config['n_dim'], config['data_size'], config['data_split'],
    #     use_cache=config['use_cache'])
    rosen = RosenData(config['n_dim'], 20000, config['data_split'], use_cache=False)
    x_pool, y_pool = rosen.dataset('pool')
    x_train, y_train = rosen.dataset('train')
    x_val, y_val = rosen.dataset('train')

    set_random(config['random_seed'])
    model = get_model(config['retrain'], config['model_path'], (x_train, y_train), (x_val, y_val))
    estimator = build_estimator(config['estimator'], model)

    estimation = estimator.estimate(x_pool, x_train, y_train)

    vae = get_vae(restore=False, rosen=rosen, patience=patience)

    # Check vae
    x_batch = x_train[:30]
    decoded = vae.predict(x_batch)
    for i in range(5):
        plt.scatter([x[i] for x in x_batch], [y[i] for y in decoded])
    plt.show()


    # # Make 3d picture of pool points
    # points = 500
    # x_batch = x_pool[:points]
    # vae.eval()
    # with torch.no_grad():
    #     surface = vae(torch.DoubleTensor(x_batch))[1].tolist()
    # xs = [point[0] for point in surface]
    # ys = [point[1] for point in surface]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # Axes3D(fig).scatter(xs, ys, zs=y_pool[:points])
    # plt.show()



















