import sys
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

import torch

from fastai.vision import (rand_pad, flip_lr, ImageDataBunch, Learner, accuracy, Image)
from fastai.callbacks import EarlyStoppingCallback

sys.path.append('..')
from model.cnn import AnotherConv, SimpleConv
from model.resnet import resnet_masked
from dataloader.builder import build_dataset
from uncertainty_estimator.masks import DEFAULT_MASKS
from experiments.utils.fastai import ImageArrayDS
from active_learning.simple_update import update_set
from experiment_setup import ROOT_DIR
from pathlib import Path
from experiments.active_learning_mnist import prepare_mnist
from experiments.active_learning import build_model, train_classifier


config = {
    'val_size': 10_000,
    'model_type': 'simple_conv',
    'batch_size': 256,
    'patience': 3,
    'epochs': 50,
    'start_lr': 5e-4,
    'weight_decay': 0.2,
}

x_train, y_train, x_val, y_val, train_tfms = prepare_mnist(config)

loss_func = torch.nn.CrossEntropyLoss()
model = build_model(config['model_type'])

learner = train_classifier(model, config, x_train, y_train, x_val, y_val)


# plt.plot((0, 1), (0, 1))
# path = Path(ROOT_DIR) / 'experiments' / 'data' / 'ood' / 'test.png'
# plt.show()







