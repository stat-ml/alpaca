import os
from os import path

import pandas as pd
import numpy as np

from experiment_setup import ROOT_DIR
from .saver import DataSaver
from sklearn.model_selection import train_test_split
import wget


URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'


class ConcreteData:
    """Load/provides boston housing dataset"""
    def __init__(self, use_cache=False, val_split=0.2):
        self.use_cache = use_cache
        cache_dir = path.join(ROOT_DIR, 'dataloader/data/concrete')
        self.saver = DataSaver(cache_dir)
        self.val_split = val_split
        self._build_dataset(cache_dir)

    def dataset(self, label):
        if self.use_cache:
            return self.saver.load(label)

        if label == 'train':
            data = self.train
        elif label == 'val':
            data = self.val
        elif label == 'all':
            data = np.concatenate(self.train, self.val)
        else:
            raise RuntimeError("Wrong label")

        x, y = data[:, :-1], data[:, -1:]
        self.saver.save(x, y, label)
        return x, y

    def _build_dataset(self, cache_dir):
        if not path.exists(cache_dir):
            os.makedirs(cache_dir)
        data_path = path.join(cache_dir, 'concrete.data')
        print(data_path)
        if not path.exists(data_path):
            wget.download(URL, data_path)
        self.df = pd.read_excel(data_path)
        table = self.df.to_numpy()
        self.train, self.val = train_test_split(table, test_size=self.val_split, shuffle=True)

