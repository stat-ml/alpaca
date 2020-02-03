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
from model.model_alternative import AnotherConv
from dataloader.builder import build_dataset
from uncertainty_estimator.bald import Bald, BaldMasked
from uncertainty_estimator.masks import build_masks, build_mask, DEFAULT_MASKS
from experiments.utils.fastai import ImageArrayDS, Inferencer


# plt.switch_backend('Qt4Agg')  # to work with remote server
# torch.cuda.set_device(1)
torch.backends.cudnn.benchmark = True


total_size = 60_000
val_size = 50_0
start_size = 5_000
step_size = 500
steps = 20
retrain = True
nn_runs = 100


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


# model = AnotherConv()
# model = resnet_masked(pretrained=True)
model = resnet_linear(pretrained=True, dropout_rate=0.5)
learner = Learner(data, model, metrics=accuracy, loss_func=loss_func)

model_path = "experiments/data/model.pt"
if retrain or not os.path.exists(model_path):
    learner.fit(10, 5e-4, wd=0.2)
    torch.save(model.state_dict(), model_path)
else:
    model.load_state_dict(torch.load(model_path))


images = torch.FloatTensor(x_val)# .to('cuda')
inferencer = Inferencer(model)

mask = build_mask('l_dpp')
estimator = BaldMasked(inferencer, dropout_mask=mask, num_classes=10, keep_runs=True, nn_runs=nn_runs)
estimations = estimator.estimate(images)



# mcd = estimator.last_mcd_runs().reshape(20, nn_runs*10)
# dpp = FiniteDPP('likelihood', L=np.corrcoef(mcd))
# idxs = set()
# while len(idxs) < step_size:
#     dpp.sample_exact()
#     idxs.update(dpp.list_of_samples[-1])
# idxs = list(idxs)[:step_size]
#
# # idxs = np.argsort(estimations)[::-1]
# print(idxs)
# print(estimations[idxs])

