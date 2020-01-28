import sys
sys.path.append('..')
from functools import partial

import torch
from torch.utils.data import Dataset

from fastai.vision import (
    rand_pad, flip_lr, ImageDataBunch, Learner, accuracy,
    simple_cnn, Image, cnn_learner)
from fastai.vision.models.wrn import wrn_22
from fastai.vision import models
from fastai.callbacks import EarlyStoppingCallback

from model.model_alternative import AnotherConv
from dataloader.builder import build_dataset
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt


plt.switch_backend('Qt4Agg')
torch.cuda.set_device(1)
torch.backends.cudnn.benchmark = True


class ImageArrayDS(Dataset):
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


dataset = build_dataset('cifar_10', val_size=10_000)
x_set, y_set = dataset.dataset('train')
x_val, y_val = dataset.dataset('val')

shape = (-1, 3, 32, 32)
x_set = ((x_set - 128)/128).reshape(shape)
x_val = ((x_val - 128)/128).reshape(shape)


train_tfms = [*rand_pad(4, 32), flip_lr(p=0.5)]
train_ds = ImageArrayDS(x_set, y_set, train_tfms)
val_ds = ImageArrayDS(x_val, y_val)

data = ImageDataBunch.create(train_ds, val_ds, bs=512)

model = AnotherConv()
# model = simple_cnn((3, 16, 16, 10))

callbacks = [
    partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.001, patience=3)]
loss_func = torch.nn.CrossEntropyLoss()
learner = Learner(data, model, metrics=accuracy, loss_func=loss_func, callback_fns=callbacks).to_fp16()
learner.fit_one_cycle(100, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)

learner.recorder.plot_losses(); plt.show()


