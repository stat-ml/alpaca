import os
import os.path as path
import numpy as np


class DataSaver:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def save(self, x, y, name=''):
        if not(path.exists(self.data_dir)):
            os.makedirs(self.data_dir)
        np.save(path.join(self.data_dir, f"x_{name}.npy"), x)
        np.save(path.join(self.data_dir, f"y_{name}.npy"), y)

    def load(self, name=''):
        x_path = path.join(self.data_dir, f"x_{name}.npy")
        y_path = path.join(self.data_dir, f"y_{name}.npy")
        assert path.exists(x_path) and path.exists(y_path)
        x_set = np.load(x_path, allow_pickle=True)
        y_set = np.load(y_path, allow_pickle=True)
        return x_set, y_set
