from os import path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import zipfile

from .config import DATA_DIR
from .saver import DataSaver
from .downloader import download


URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip'


class NavalPropulsionData:
    """Load/provides naval propulsion dataset"""
    def __init__(self, use_cache=False, val_split=0.2):
        self.use_cache = use_cache
        cache_dir = path.join(DATA_DIR, 'dataloader/data/naval_propulsion')
        self.saver = DataSaver(cache_dir)
        self.val_split = val_split
        self._build_dataset(cache_dir)

    def dataset(self, label):
        if self.use_cache:
            return self.saver.load(label)

        data = self.data[label]
        x, y = data[:, :-2], data[:, -1:]
        self.saver.save(x, y, label)
        return x, y

    def _build_dataset(self, cache_dir):
        data_path = download(cache_dir, 'naval_propulsion.zip', URL)
        with zipfile.ZipFile(data_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        file_path = path.join(cache_dir, 'UCI CBM Dataset', 'data.txt')
        self.df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        table = self.df.to_numpy()
        train, val = train_test_split(table, test_size=self.val_split, shuffle=True)
        self.data = {
            'train': train,
            'val': val,
            'all': np.concatenate((train, val))
        }


if __name__ == '__main__':
    dataset = NavalPropulsionData()
    x_train, y_train = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')
    print(x_train.shape, y_val.shape)

