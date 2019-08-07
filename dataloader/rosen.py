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
            self, n_train, n_val, n_test, n_pool, n_dim,
            cache_dir='dataloader/data/rosen', rewrite=False):
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.n_pool = n_pool
        self.n_dim = n_dim
        self.cache_dir = cache_dir
        self.rewrite = rewrite

    def dataset(self, use_cache=False):
        if use_cache:
            try:
                return self._load()
            except AssertionError:
                print("Can't use cache, generating new dataset")

        X_train, y_train = self._rosen_set('train')
        X_val, y_val = self._rosen_set('val')
        X_test, y_test = self._rosen_set('test')
        X_pool, y_pool = self._rosen_set('pool')

        return X_train, y_train, X_val, y_val, X_test, y_test, X_pool, y_pool

    def _load(self):
        datasets = []
        labels = ['train', 'val', 'test', 'pool']
        for label in labels:
            x_path = path.join(self.cache_dir, f'x_{label}.npy')
            y_path = path.join(self.cache_dir, f'y_{label}.npy')
            assert path.exists(x_path)
            assert path.exists(y_path)

            x_set = np.load(x_path)
            y_set = np.load(y_path)
            assert x_set.shape == (getattr(self, 'n_' + label), self.n_dim)
            assert y_set.shape == (getattr(self, 'n_' + label), 1)

            datasets.append(x_set)
            datasets.append(y_set)

        return datasets

    def _rosen_set(self, label):
        x_shape = (getattr(self, 'n_'+label), self.n_dim)
        X = np.random.random(x_shape)
        y = self._rosen(X)
        if self.rewrite:
            self._save('x_'+label, X)
            self._save('y_'+label, y)
        return X, y

    def _save(self, name, dataset):
        if not(path.exists(self.cache_dir)):
            os.makedirs(self.cache_dir)
        np.save(path.join(self.cache_dir, name + '.npy'), dataset)

    @staticmethod
    def _rosen(x):
        return rosen(x.T)[:, None]


if __name__ == '__main__':
    X_train, y_train, _, _, _, _, _, _ = RosenData(200, 200, 200, 1000, 10).dataset(True)
    print(y_train[:2])
