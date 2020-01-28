import sys
sys.path.append('..')
import torch

from torch.utils.data import TensorDataset, Dataset

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

shape = (-1, 3, 32, 32)
x_set = ((x_set - 128)/128).reshape(shape)
x_val = ((x_val - 128)/128).reshape(shape)

x = np.concatenate((x_set, x_val), axis=0)
y = np.concatenate((y_set, y_val))
train_idxs = range(len(x_set))
val_idxs = range(len(x_set), len(x_set) + len(x_val))

train_ds = TensorDataset(torch.Tensor(x_set).view(-1, 3, 32, 32), torch.LongTensor(y_set))
val_ds = TensorDataset(torch.Tensor(x_val).view(-1, 3, 32, 32), torch.LongTensor(y_val))


class ImageArrayDS:
    def __init__(self, images, labels, tfms=None):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
        self.tfms = tfms

    def __getitem__(self, idx):
        image = Image(self.images[idx])
        if self.tfms is not None:
            image = image.apply_tfms(self.tfms)
        return image, self.labels[idx]

    def __len__(self):
        return len(self.images)

train_tfms = [*rand_pad(4, 32), flip_lr(p=0.5)]
train_ds = ImageArrayDS(x_set, y_set, train_tfms)
val_ds = ImageArrayDS(x_val, y_val)
# train_ds = TensorDataset(torch.Tensor(x_set).view(-1, 3, 32, 32), torch.LongTensor(y_set))
# val_ds = TensorDataset(torch.Tensor(x_val).view(-1, 3, 32, 32), torch.LongTensor(y_val))

data = ImageDataBunch.create(train_ds, val_ds, bs=512)
# ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
# for tf in ds_tfms[0]:
#     data.add_tfm(tf)
# import pdb; pdb.set_trace()

# print(type(data))
# data = (ArrayImageList.from_numpy(x)
#         .split_by_idxs(train_idxs, val_idxs)
#         .label_from_array(y)
#         .databunch(bs=128))
#

model = AnotherConv()
# model = simple_cnn((3, 16, 16, 10))


loss_func = torch.nn.CrossEntropyLoss()
learn = Learner(data, model, metrics=accuracy, loss_func=loss_func).to_fp16()
learn.fit_one_cycle(40, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)

