import os
import sys
sys.path.append('..')

import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

from fastai.vision import rand_pad, flip_lr, ImageDataBunch, Learner, accuracy, Image, create_cnn_model
from fastai.vision import models
from fastai.basic_data import DataBunch
from dppy.finite_dpps import FiniteDPP

from model.resnet import resnet_masked, resnet_linear
from model.cnn import AnotherConv
from model.dense import Dense
from dataloader.builder import build_dataset
from experiment_setup import build_estimator
from uncertainty_estimator.masks import build_masks, build_mask, DEFAULT_MASKS
from experiments.utils.fastai import ImageArrayDS, Inferencer
from experiments.active_learning import update_set


border = 1.2
xx, yy = np.meshgrid(np.linspace(-border, border, 100), np.linspace(-border, border, 100))
x_val = np.vstack((xx.ravel(), yy.ravel())).T




def main():
    for noise_level in [0, 0.1, 0.2, 0.3]:
        x, y = xor(500, noise_level)
        model = train(x, y)
        eval(model, x, y)



def xor(points, noise_level=0.):
    rng = np.random
    x = 2*rng.random((points, 2)) - 1
    noised_x = x + noise_level * rng.randn(points, 2)
    y = np.logical_xor(noised_x[:, 0] > 0, noised_x[:, 1] > 0)
    return x, y


def train(x, y):
    loss_func = torch.nn.CrossEntropyLoss()
    model = Dense((2, 100, 100, 100, 2), dropout_rate=0.5)

    train_ds = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))
    data = DataBunch.create(train_ds, train_ds, bs=10)

    learner = Learner(data, model, metrics=accuracy, loss_func=loss_func)

    learner.fit_one_cycle(50)
    return model

def eval(model, x, y, method='basic_bern'):
    t_val = torch.FloatTensor(x_val).cuda()
    mask = build_mask('basic_bern')
    estimator = build_estimator('bald_masked', model, dropout_mask=mask, num_classes=2)
    estimations = sigmoid(estimator.estimate(t_val))

    # estimations = model(t_val)[:, 0]
    Z = estimations.reshape(xx.shape)
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, cmap=plt.cm.gray, edgecolors=(0, 0, 0))
    plt.imshow(Z, interpolation='nearest',
                           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                           aspect='auto', origin='lower', cmap=plt.cm.RdBu_r)
    plt.show()


if __name__ == '__main__':
    main()
