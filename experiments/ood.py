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
from sklearn.metrics import roc_auc_score


config = {
    'val_size': 10_000,
    'model_type': 'simple_conv',
    'batch_size': 256,
    'patience': 3,
    'epochs': 50,
    'start_lr': 5e-4,
    'weight_decay': 0.2,
    'reload': False
}




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
images = torch.FloatTensor(x_val)# .to('cuda')
inferencer = Inferencer(model)

probabilities = F.softmax(inferencer(images), dim=1).detach().cpu().numpy()
predictions = np.argmax(probabilities, axis=-1)

mistake = 1 - (predictions == y_val).astype(np.int)
# Uncertainty estimation as 1 - p_max
ue = 1 - probabilities[np.arange(len(probabilities)), np.argmax(probabilities, axis=-1)]


print(roc_auc_score(mistake, ue))


# mask = build_mask('k_dpp')
# estimator = BaldMasked(inferencer, dropout_mask=mask, num_classes=10, nn_runs=nn_runs)
# estimations = estimator.estimate(images)
# print(estimations)


