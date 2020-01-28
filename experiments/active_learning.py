import sys
sys.path.append('..')
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
from uncertainty_estimator.bald import Bald


plt.switch_backend('Qt4Agg')
torch.cuda.set_device(1)
torch.backends.cudnn.benchmark = True


val_size = 10_000
pool_size = 20_000
start_size = 5_000
step_size = 500
steps = 20


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


def main():
    # Load data
    dataset = build_dataset('cifar_10', val_size=10_000)
    x_set, y_set = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')

    shape = (-1, 3, 32, 32)
    x_set = ((x_set - 128)/128).reshape(shape)
    x_val = ((x_val - 128)/128).reshape(shape)

    # Start data split
    x_pool, x_train, y_pool, y_train = train_test_split(x_set, y_set, test_size=start_size, stratify=y_set)

    train_tfms = [*rand_pad(4, 32), flip_lr(p=0.5)]
    loss_func = torch.nn.CrossEntropyLoss()
    model = AnotherConv()
    # model = simple_cnn((3, 16, 16, 10))

    # Active learning
    val_accuracy = []

    for i in range(steps):
        print(f"Epoch {i+1}, train size: {len(x_train)}")
        train_ds = ImageArrayDS(x_train, y_train, train_tfms)
        val_ds = ImageArrayDS(x_val, y_val)

        data = ImageDataBunch.create(train_ds, val_ds, bs=256)

        # callbacks = [partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.001, patience=3)]
        callbacks = []
        learner = Learner(data, model, metrics=accuracy, loss_func=loss_func, callback_fns=callbacks) #.to_fp16()
        # learner.fit_one_cycle(100, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)
        learner.fit(2, 3e-3, wd=0.2)

        val_accuracy.append(learner.recorder.metrics[-1][0].item())

        x_pool, x_train, y_pool, y_train = update_set(
            x_pool, x_train, y_pool, y_train, method='mcdue', model=model)

    # Display results
    plot_metric(val_accuracy)


def update_set(x_pool, x_train, y_pool, y_train, method='mcdue', model=None):
    if method == 'random':
        idxs = range(step_size)
    elif method == 'mcdue':
        images = torch.FloatTensor(x_pool).to('cuda')
        estimator = Bald(model, num_classes=10, nn_runs=100)
        estimations = estimator.estimate(images)
        idxs = np.argsort(estimations)[::-1][:step_size]
    x_add, y_add = np.copy(x_pool[idxs]), np.copy(y_pool[idxs])
    x_train = np.concatenate((x_train, x_add))
    y_train = np.concatenate((y_train, y_add))
    x_pool = np.delete(x_pool, idxs, axis=0)
    y_pool = np.delete(y_pool, idxs, axis=0)
    return x_pool, x_train, y_pool, y_train


def plot_metric(metric, title=None):
    plt.figure(figsize=(16, 9))
    title = title or f"Validation accuracy, start size {start_size}, step size {step_size}"
    plt.title(title)
    plt.plot(metric, label='random')
    plt.xlabel("Steps")
    plt.ylabel("Accuracy on validation")
    plt.legend(loc='upper left')
    plt.show()



if __name__ == '__main__':
    main()
