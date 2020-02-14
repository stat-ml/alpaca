import os
import sys
sys.path.append('..')

import torch
import torch.nn.functional as F
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
from active_learning.simple_update import update_set, entropy


# == Place to develop, experiment and debug new methods of active learning == #

# torch.cuda.set_device(1)


total_size = 60_000
val_size = 10_000
start_size = 1_000
step_size = 10
steps = 8
reload = True
nn_runs = 100

pool_size = 200


# Load data
dataset = build_dataset('cifar_10', val_size=val_size)
x_set, y_set = dataset.dataset('train')
x_val, y_val = dataset.dataset('val')

shape = (-1, 3, 32, 32)
x_set = ((x_set - 128)/128).reshape(shape)
x_val = ((x_val - 128)/128).reshape(shape)

# x_pool, x_train, y_pool, y_train = train_test_split(x_set, y_set, test_size=start_size, stratify=y_set)
x_train, y_train = x_set,  y_set

train_tfms = [*rand_pad(4, 32), flip_lr(p=0.5)]
train_ds = ImageArrayDS(x_train, y_train, train_tfms)
val_ds = ImageArrayDS(x_val, y_val)
data = ImageDataBunch.create(train_ds, val_ds, bs=256)


loss_func = torch.nn.CrossEntropyLoss()

np.set_printoptions(threshold=sys.maxsize, suppress=True)

model = AnotherConv()
# model = resnet_masked(pretrained=True)
# model = resnet_linear(pretrained=True, dropout_rate=0.5, freeze=False)

# learner = Learner(data, model, metrics=accuracy, loss_func=loss_func)
#
# model_path = "experiments/data/model.pt"
# if reload and os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path))
# else:
#     learner.fit(10, 1e-3, wd=0.02)
#     torch.save(model.state_dict(), model_path)
#
# images = torch.FloatTensor(x_val)# .to('cuda')
#
# inferencer = Inferencer(model)
# predictions = F.softmax(inferencer(images), dim=1).detach().cpu().numpy()[:10]


repeats = 3
methods = ['goy', 'mus', 'cosher']
from random import random

results = []
for _ in range(repeats):
    start = 0.1 * random()
    for method in methods:
        accuracies = [start]
        current = start
        for i in range(10):
            current += 0.1*random()
            accuracies.append(current)
        records = list(zip(accuracies, range(len(accuracies)), [method] * len(accuracies)))
        results.extend(records)

import pandas as pd
import seaborn as sns
df = pd.DataFrame(results, columns=['accuracy', 'step', 'method'])
sns.lineplot('step', 'accuracy', hue='method', data=df)
plt.show()












# idxs = np.argsort(entropies)[::-1][:10]
# print(idxs)

# mask = build_mask('k_dpp')
# estimator = BaldMasked(inferencer, dropout_mask=mask, num_classes=10, nn_runs=nn_runs)
# estimations = estimator.estimate(images)
# print(estimations)









