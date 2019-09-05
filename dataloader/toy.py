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
        if label == 'train':
            x = uniform(x0, x1, points)
            y = np.power(x, 3) + normal(0, self.noise, points)
        elif label == 'val':
            x = np.arange(k*x0, k*x1, 0.1)
            y = np.power(x, 3)

        self.saver.save(x, y, label)
        return x, y


class ToySinData:
    def __init__(self, noise=0.02, use_cache=False):
        self.noise = noise
        self.use_cache = use_cache
        self.saver = DataSaver('dataloader/data/toy_sin')

    def dataset(self, label):
        points = 20

        if self.use_cache:
            return self.saver.load(label)
        if label == 'train':
            x = np.concatenate((uniform(-0.8, -0.3, points), uniform(0.6, 1.1, points)))
            # x = uniform(-1, 1, 2*points)
            y = self._function(x) + normal(0, self.noise, 2*points)
        elif label == 'val':
            x = np.arange(-1.5, 1.5, 0.01)
            y = self._function(x)

        self.saver.save(x, y, label)
        return x, y

    def _function(self, x):
        return 0.3*x + 0.3*np.sin(2*np.pi*x) + 0.3*np.sin(4*np.pi*x)



