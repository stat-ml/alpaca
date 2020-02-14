import sys
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

import torch
from fastai.vision import (ImageDataBunch, Learner, accuracy)
from fastai.callbacks import EarlyStoppingCallback

sys.path.append('..')
from dataloader.builder import build_dataset
from experiments.utils.fastai import ImageArrayDS
from active_learning.simple_update import update_set
from uncertainty_estimator.masks import DEFAULT_MASKS
from experiments.active_learning import main

torch.cuda.set_device(1)
torch.backends.cudnn.benchmark = True



def prepare_mnist(config):
    dataset = build_dataset('mnist', val_size=config['val_size'])
    x_set, y_set = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')

    shape = (-1, 1, 28, 28)
    x_set = ((x_set - 128) / 128).reshape(shape)
    x_val = ((x_val - 128) / 128).reshape(shape)

    train_tfms = []

    return x_set, y_set, x_val, y_val, train_tfms


mnist_config = {
    'repeats': 5,
    'start_size': 100,
    'step_size': 20,
    'val_size': 10_000,
    'pool_size': 10_000,
    'steps': 30,
    'methods': ['random', 'error_oracle', 'max_entropy', *DEFAULT_MASKS],
    'epochs_per_step': 30,
    'patience': 2,
    'model_type': 'simple_conv',
    'nn_runs': 100,
    'batch_size': 32,
    'start_lr': 5e-4,
    'weight_decay': 0.2,
    'prepare_dataset': prepare_mnist
}


if __name__ == '__main__':
    main(mnist_config)
