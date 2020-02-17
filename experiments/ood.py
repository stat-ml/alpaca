import os
import sys
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

import torch
import torch.nn.functional as F

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
from experiments.utils.fastai import Inferencer
from sklearn.metrics import roc_auc_score, roc_curve
from uncertainty_estimator.masks import build_mask
from experiment_setup import build_estimator
from active_learning.simple_update import entropy


config = {
    'val_size': 10_000,
    'model_type': 'simple_conv',
    'batch_size': 256,
    'patience': 3,
    'epochs': 50,
    'start_lr': 5e-4,
    'weight_decay': 0.2,
    'reload': False,
    'nn_runs': 100,
    'estimators': ['max_prob', 'max_entropy', *DEFAULT_MASKS],
    'repeats': 1
}


def benchmark_uncertainty():
    x_train, y_train, x_val, y_val, train_tfms = prepare_mnist(config)

    train_ds = ImageArrayDS(x_train, y_train, train_tfms)
    val_ds = ImageArrayDS(x_val, y_val)
    data = ImageDataBunch.create(train_ds, val_ds, bs=256)

    loss_func = torch.nn.CrossEntropyLoss()
    np.set_printoptions(threshold=sys.maxsize, suppress=True)
    model = SimpleConv()

    learner = Learner(data, model, metrics=accuracy, loss_func=loss_func)
    #
    model_path = "experiments/data/model.pt"
    if config['reload'] and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        learner.fit(10, 1e-3, wd=0.02)
        torch.save(model.state_dict(), model_path)
    images = torch.FloatTensor(x_val).cuda()

    probabilities = F.softmax(model(images), dim=1).detach().cpu().numpy()
    predictions = np.argmax(probabilities, axis=-1)

    results = []
    plt.figure(figsize=(10, 8))
    for i in range(config['repeats']):
        for name in config['estimators']:
            ue = calc_ue(model, images, probabilities, name, config['nn_runs'])
            mistake = 1 - (predictions == y_val).astype(np.int)

            roc_auc = roc_auc_score(mistake, ue)
            print(name, roc_auc)
            results.append((name, roc_auc))

            if i == config['repeats'] - 1:
                fpr, tpr, thresholds = roc_curve(mistake, ue, pos_label=1)
                plt.plot(fpr, tpr, label=name, alpha=0.8)

    plt.title('MNIST uncertainty ROC')
    plt.legend()
    plt.show()


def calc_ue(model, images, probabilities, estimator_type='max_prob', nn_runs=100):
    # Uncertainty estimation as 1 - p_max
    if estimator_type == 'max_prob':
        ue = 1 - probabilities[np.arange(len(probabilities)), np.argmax(probabilities, axis=-1)]
    elif estimator_type == 'max_entropy':
        ue = entropy(probabilities)
    else:
        mask = build_mask(estimator_type)
        estimator = build_estimator('bald_masked', model, dropout_mask=mask, num_classes=10, nn_runs=nn_runs)
        ue = estimator.estimate(images)
        print(ue[:10])

    return ue


if __name__ == '__main__':
    benchmark_uncertainty()

# print(estimations)

