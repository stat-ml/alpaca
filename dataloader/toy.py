import numpy as np
from numpy.random import uniform, normal
from .saver import DataSaver


class ToyQubicData:
    """Generate noisy points of x*3 on [-4, 4]"""
    def __init__(self, noise=9, use_cache=False):
        self.noise = noise
        self.use_cache = use_cache
        self.saver = DataSaver('dataloader/data/toy_qubic')

    def dataset(self, label):
        points = 20
        x0, x1 = -4, 4
        k = 1.5

        if self.use_cache:
            return self.saver.load(label)
        if label in ['train', 'val']:
            x = uniform(x0, x1, points)
            y = np.power(x, 3) + normal(0, self.noise, points)
        elif label == 'ground_truth':
            x = np.arange(k*x0, k*x1, 0.2)
            y = np.power(x, 3)

        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        self.saver.save(x, y, label)
        return x, y


class ToySinData:
    def __init__(self, noise=0.02, use_cache=False):
        self.noise = noise
        self.use_cache = use_cache
        self.saver = DataSaver('dataloader/data/toy_sin')

    def dataset(self, label):
        points = 30

        if self.use_cache:
            return self.saver.load(label)
        if label in ['train', 'val']:
            x = np.concatenate((uniform(-1, -0.3, points), uniform(0.5, 1.2, points)))
            # x = np.concatenate((uniform(-4, -3.3, points), uniform(3.2, 4, points)))
            y = self._function(x) + normal(0, self.noise, 2*points)
        elif label == 'ground_truth':
            x = np.arange(-1.5, 1.5, 0.04)
            # x = np.arange(-5, 5, 0.04)
            y = self._function(x)

        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        self.saver.save(x, y, label)
        return x, y

    def _function(self, x):
        return 0.3*x + 0.3*np.sin(2*np.pi*x) + 0.3*np.sin(4*np.pi*x)



