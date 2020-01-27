import sys
sys.path.append('..')
import torch

from torch.utils.data import TensorDataset

from fastai.vision import (
    untar_data, rand_pad, flip_lr, ImageDataBunch, Learner, accuracy,
    cifar_stats, URLs, simple_cnn)
from fastai.vision.models.wrn import wrn_22
from fastai.vision.models import resnet18, resnet34, resnet50
from fastai.basic_data import DataBunch

from model.model_alternative import AnotherConv
from dataloader.builder import build_dataset
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt


plt.switch_backend('Qt4Agg')
torch.cuda.set_device(1)

loss_func = torch.nn.CrossEntropyLoss()


# dataset = build_dataset('cifar_10', val_size=10_000)
# x_set, y_set = dataset.dataset('train')
# x_val, y_val = dataset.dataset('val')
#
# x_set = np.array(x_set - 128)/128
# x_val = (x_val - 128)/128
#
# # torch.backends.cudnn.benchmark = True
# # # path = untar_data(URLs.CIFAR)
# # #
# # # # ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
# # # data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=512).normalize(cifar_stats)
# # data = ImageDataBunch.from_folder(path, valid='test', bs=512).normalize(cifar_stats)
#
# train_ds = TensorDataset(torch.Tensor(x_set).view(-1, 3, 32, 32), torch.LongTensor(y_set))
# val_ds = TensorDataset(torch.Tensor(x_val).view(-1, 3, 32, 32), torch.LongTensor(y_val))
#
# data = DataBunch.create(train_ds, val_ds, bs=512)
#

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalization
mean = X_train.mean()
std = X_train.std()
X_train = (X_train-mean)/std
X_valid = (X_valid-mean)/std

# Numpy to Torch Tensor
X_train = torch.from_numpy(np.float32(X_train)).view(-1, 1, 28, 28)
y_train = torch.from_numpy(y_train.astype(np.long))
X_valid = torch.from_numpy(np.float32(X_valid)).view(-1, 1, 28, 28)
y_valid = torch.from_numpy(y_valid.astype(np.long))

train = torch.utils.data.TensorDataset(X_train, y_train)
valid = torch.utils.data.TensorDataset(X_valid, y_valid)

data = ImageDataBunch.create(train_ds=train, valid_ds=valid)

# model = AnotherConv()
model = simple_cnn((1, 16, 16, 10))
learn = Learner(data, model, metrics=accuracy, loss_func=loss_func).to_fp16()
learn.fit_one_cycle(50, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)


