import os
import sys
sys.path.append('..')

import torch
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from fastai.vision import rand_pad, flip_lr, ImageDataBunch, Learner, accuracy, Image, create_cnn_model
from fastai.vision import models
from dppy.finite_dpps import FiniteDPP

from model.resnet import resnet_masked, resnet_linear
from model.cnn import AnotherConv
from dataloader.builder import build_dataset
from uncertainty_estimator.bald import Bald, BaldMasked
from uncertainty_estimator.masks import build_masks, build_mask, DEFAULT_MASKS
from experiments.utils.fastai import ImageArrayDS, Inferencer
from experiments.active_learning import update_set



xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
                     np.linspace(-3, 3, 50))

def xor(points, noise_level=0):
    rng = np.random
    x = 2*rng.random((points, 2)) - 1
    noised_x = x + noise_level * (2*rng.random((points, 2))-1)
    y = np.logical_xor(noised_x[:, 0] > 0, noised_x[:, 1] > 0)
    return x, y


for noise_level in [0, 0.1, 0.2, 0.3]:
    x, y = xor(500, noise_level)
    plt.figure(figsize=(7, 6))
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, cmap=plt.cm.gray, edgecolors=(0, 0, 0))
    plt.show()

