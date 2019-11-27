import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import softmax as softmax
import matplotlib.pyplot as plt

from dataloader.builder import build_dataset
from experiment_setup import build_estimator
from uncertainty_estimator.masks import build_masks, DEFAULT_MASKS
from analysis.metrics import uq_ndcg

from model.cnn import SimpleConv, MediumConv, StrongConv
from model.trainer import Trainer, EnsembleTrainer
from active_learning.al_trainer import ALTrainer

torch.cuda.set_device(1)
# %%

model_setups = {
    'mnist': {
        'model_class': SimpleConv,
        'train_samples': 5000,
        'epochs': 5,
        'batch_size': 256,
        'log_interval': 10,
        'lr': 1e-2,
        'num_classes': 10
    },
    'cifar_10': {
        'model_class': StrongConv,
        'train_samples': 45_000,
        'epochs': 50,
        'batch_size': 256,
        'log_interval': 150,
        'lr': 1e-2,
        'num_classes': 9,
    }
}

config = {
    'use_cuda': True,
    'seed': 1,

    'nn_runs': 150,
    'patience': 5,
    'dropout_uq': 0.5,

    'n_models': 10,

    # 'dataset': 'mnist',
    'dataset': 'cifar_10',

    'model_runs': 3,
    'repeat_runs': 3,

    'al_start': 500,
    'al_step': 200
}

config.update(model_setups[config['dataset']])

# %% md

#### Load data and preprocess
# %%
dataset = build_dataset(config['dataset'], val_size=10_000)
x_train, y_train = dataset.dataset('train')
x_val, y_val = dataset.dataset('val')
x_train, x_pool, y_train, y_pool = train_test_split(
    x_train, y_train, train_size=config['al_start'], stratify=y_train)


# %%
def scale(images):
    return (images - 128) / 128


x_train = scale(x_train)
x_val = scale(x_val)
x_pool = scale(x_pool)
# %%
if config['dataset'] == 'mnist':
    input_shape = (-1, 1, 28, 28)
elif config['dataset'] == 'cifar_10':
    input_shape = (-1, 3, 32, 32)
x_train = x_train.reshape(input_shape)
x_val = x_val.reshape(input_shape)
x_pool = x_pool.reshape(input_shape)

y_train = y_train.astype('long').reshape(-1)
y_pool = y_pool.astype('long').reshape(-1)
y_val = y_val.astype('long').reshape(-1)

# %% md
#### Train model
# %%
masks = build_masks(DEFAULT_MASKS)
model = config['model_class']()
trainer = Trainer(model)
# %%
mask = masks['basic_bern']
estimator = build_estimator(
    'bald_masked', model, nn_runs=config['nn_runs'], dropout_mask=mask,
    dropout_rate=config['dropout_uq'], num_classes=config['num_classes'])
# %%
trainer.fit((x_train, y_train), (x_val, y_val))
# active_teacher = ALTrainer(trainer, estimator, y_pool=y_pool)
# errors = active_teacher.train(x_train, y_train, x_val, y_val, x_pool)

