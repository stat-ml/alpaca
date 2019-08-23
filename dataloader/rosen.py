import os
import os.path as path

from scipy.optimize import rosen
import numpy as np


class RosenData:
    """
    Generate points for n_dim Rosenbrock function
    Or loads the points from cached version
    """
    def __init__(
            self, n_dim, data_size, data_split=None,
            use_cache=True, cache_dir='dataloader/data/rosen', rewrite=False):

        self.n_dim = n_dim
        self.data_size = data_size
        self.data_split = data_split
        self.cache_dir = cache_dir
        self.splits = self._parse_split(data_split)

        self.x_set, self.y_set = self._build_set(use_cache, rewrite)

    def dataset(self, label):
        assert label in ['train', 'val', 'test', 'pool']
        index = self.splits[label]
        return self.x_set[index[0]:index[1]], self.y_set[index[0]:index[1]]

    def _build_set(self, use_cache, rewrite):
        if use_cache:
            try:
                return self._load()
            except AssertionError:
                print("Can't use cache, generating new dataset")

        x = 2*np.random.random((self.data_size, self.n_dim)) - 1
        y = rosen(x.T)[:, None]
        if rewrite:
            self._save('x', x)
            self._save('y', y)
        return x, y

    def _save(self, name, dataset):
        if not(path.exists(self.cache_dir)):
            os.makedirs(self.cache_dir)
        np.save(path.join(self.cache_dir, name + '.npy'), dataset)

    def _load(self):
        x_path = path.join(self.cache_dir, 'x.npy')
        y_path = path.join(self.cache_dir, 'y.npy')
        assert path.exists(x_path) and path.exists(y_path)
        x_set = np.load(x_path)
        y_set = np.load(y_path)
        assert x_set.shape == (self.data_size, self.n_dim)

        return x_set, y_set

    def _parse_split(self, splits):
        assert abs(sum(splits) - 1) < 1e-6  # split should sum to 1
        indices = [round(self.data_size*sum(splits[:i])) for i in range(4)]

        return {
            'train': (indices[0], indices[1]),
            'val': (indices[1], indices[2]),
            'test': (indices[2], indices[3]),
            'pool': (indices[3], self.data_size),
        }
