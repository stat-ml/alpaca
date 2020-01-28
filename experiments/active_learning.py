import sys
sys.path.append('..')
import torch

from torch.utils.data import TensorDataset

from fastai.vision import (
    untar_data, rand_pad, flip_lr, ImageDataBunch, Learner, accuracy,
    cifar_stats, URLs, simple_cnn, Image, ImageList, cnn_learner)
from fastai.vision.models.wrn import wrn_22
from fastai.basic_data import DataBunch
from fastai.vision import models

from model.model_alternative import AnotherConv
from dataloader.builder import build_dataset
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt


plt.switch_backend('Qt4Agg')
torch.cuda.set_device(1)
torch.backends.cudnn.benchmark = True


class ArrayImageList(ImageList):
    @classmethod
    def from_numpy(cls, numpy_array):
        return cls(items=numpy_array)

    def label_from_array(self, array, label_cls=None, **kwargs):
        return self._label_from_list(array, label_cls=label_cls, **kwargs)

    def get(self, i):
        n = self.items[i]
        n = torch.tensor(n).float()
        return Image(n)


dataset = build_dataset('cifar_10', val_size=10_000)
x_set, y_set = dataset.dataset('train')
x_val, y_val = dataset.dataset('val')

x_set = ((x_set - 128)/128)
x_val = ((x_val - 128)/128)

x = np.concatenate((x_set, x_val)).reshape((-1, 3, 32, 32))
y = np.concatenate((y_set, y_val))
train_idxs = range(len(x_set))
val_idxs = range(len(x_set), len(x_set) + len(x_val))

train_ds = TensorDataset(torch.Tensor(x_set).view(-1, 3, 32, 32), torch.LongTensor(y_set))
val_ds = TensorDataset(torch.Tensor(x_val).view(-1, 3, 32, 32), torch.LongTensor(y_val))

data = ImageDataBunch.create(train_ds, val_ds, bs=512)
# print(type(data))
#

# data = (ArrayImageList.from_numpy(x)
#         .split_by_idxs(train_idxs, val_idxs)
#         .label_from_array(y)
#         .databunch(bs=128))
#

model = AnotherConv()
# model = simple_cnn((3, 16, 16, 10))

loss_func = torch.nn.CrossEntropyLoss()
learn = Learner(data, model, metrics=accuracy, loss_func=loss_func).to_fp16()
learn.fit_one_cycle(20, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)

