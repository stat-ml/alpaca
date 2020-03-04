from os import path

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import numpy as np

from .config import DATA_DIR
from .saver import DataSaver


class Openml:
    def __init__(self, name, val_size=0.2):
        cache_dir = path.join(DATA_DIR, f'dataloader/data/{name}')
        self.saver = DataSaver(cache_dir)
        self.val_size = val_size
        self.name = name
        self._build_dataset(cache_dir)

    def dataset(self, label):
        return self.data[label]

    def _build_dataset(self, cache_dir):
        if not path.exists(cache_dir):
            x, y = fetch_openml(self.name, return_X_y=True, cache=True)
            y = y.astype(np.int)
            self.saver.save(x, y)
        else:
            x, y = self.saver.load()

        if self.val_size != 0:
            x_train, x_val, y_train, y_val = train_test_split(
                x, y, test_size=self.val_size, shuffle=True, stratify=y)
        else:
            x_train, y_train = x, y
            x_val, y_val = [], []

        self.data = {
            'train': (x_train, y_train),
            'val': (x_val, y_val),
        }


class MnistData(Openml):
    def __init__(self, **kwargs):
        super().__init__('mnist_784', **kwargs)


class FashionMnistData(Openml):
    def __init__(self, **kwargs):
        super().__init__('Fashion-MNIST', **kwargs)


class Cifar10(Openml):
    def __init__(self, **kwargs):
        super().__init__('CIFAR_10', **kwargs)


class SVHN(Openml):
    def __init__(self, **kwargs):
        super().__init__('SVHN', **kwargs)
