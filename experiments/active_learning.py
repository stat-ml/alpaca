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


# torch.cuda.set_device(1)
torch.backends.cudnn.benchmark = True


def prepare_cifar(config):
    dataset = build_dataset('cifar_10', val_size=config['val_size'])
    x_set, y_set = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')

    shape = (-1, 3, 32, 32)
    x_set = ((x_set - 128) / 128).reshape(shape)
    x_val = ((x_val - 128) / 128).reshape(shape)

    train_tfms = [*rand_pad(4, 32), flip_lr(p=0.5)]  # Transformation to augment images

    return x_set, y_set, x_val, y_val, train_tfms


cifar_config = {
    'val_size': 10_000,
    'pool_size': 5_000,
    'start_size': 1_000,
    'step_size': 50,
    'steps': 30,
    'methods': ['random', 'error_oracle', 'max_entropy', *DEFAULT_MASKS],
    'epochs_per_step': 20,
    'patience': 2,
    'model_type': 'conv',
    'repeats': 1,
    'nn_runs': 100,
    'batch_size': 256,
    'start_lr': 5e-4,
    'weight_decay': 0.2,
    'prepare_dataset': prepare_cifar
}


def main(config):
    # Load data
    x_set, y_set, x_val, y_val, train_tfms = config['prepare_dataset'](config)

    val_accuracy = []
    for _ in range(config['repeats']):  # more repeats for robust results
        # Start data split
        x_set, x_train_init, y_set, y_train_init = train_test_split(x_set, y_set, test_size=config['start_size'], stratify=y_set)
        _, x_pool_init, _, y_pool_init = train_test_split(x_set, y_set, test_size=config['pool_size'], stratify=y_set)

        loss_func = torch.nn.CrossEntropyLoss()

        # Active learning
        for method in config['methods']:
            print(f"== {method} ==")
            x_pool, y_pool = np.copy(x_pool_init), np.copy(y_pool_init)
            x_train, y_train = np.copy(x_train_init), np.copy(y_train_init)

            model = build_model(config['model_type'])
            accuracies = []

            for i in range(config['steps']):
                print(f"Step {i+1}, train size: {len(x_train)}")
                train_ds = ImageArrayDS(x_train, y_train, train_tfms)
                val_ds = ImageArrayDS(x_val, y_val)
                data = ImageDataBunch.create(train_ds, val_ds, bs=config['batch_size'])

                callbacks = [partial(EarlyStoppingCallback, min_delta=1e-3, patience=config['patience'])]
                learner = Learner(data, model, metrics=accuracy, loss_func=loss_func, callback_fns=callbacks)
                learner.fit(config['epochs_per_step'], config['start_lr'], wd=config['weight_decay'])

                if i != config['steps'] - 1:
                    x_pool, x_train, y_pool, y_train = update_set(
                        x_pool, x_train, y_pool, y_train, config['step_size'], method=method, model=model)

                accuracies.append(learner.recorder.metrics[-1][0].item())

            records = list(zip(accuracies, range(len(accuracies)), [method] * len(accuracies)))
            val_accuracy.extend(records)

    # Display results
    plot_metric(val_accuracy, config)


def plot_metric(metrics, config, title=None):
    plt.figure(figsize=(16, 9))
    title = title or f"Validation accuracy, start size {config['start_size']}, step size {config['step_size']}, model {config['model_type']}"
    plt.title(title)

    df = pd.DataFrame(metrics, columns=['accuracy', 'step', 'method'])
    sns.lineplot('step', 'accuracy', hue='method', data=df)
    plt.show()
    plt.legend(loc='upper left')


def build_model(model_type):
    if model_type == 'conv':
        model = AnotherConv()
    elif model_type == 'resnet':
        model = resnet_masked(pretrained=True)
    elif model_type == 'simple_conv':
        model = SimpleConv()
    return model


if __name__ == '__main__':
    main(cifar_config)
