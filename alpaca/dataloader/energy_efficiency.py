from os import path

import pandas as pd
import numpy as np

from .config import DATA_DIR
from .saver import DataSaver
from .downloader import download
from sklearn.model_selection import train_test_split


URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'


# TODO: use both output features
class EnergyEfficiencyData:
    """Load/provides boston housing dataset"""
    def __init__(self, use_cache=False, val_split=0.2):
        self.use_cache = use_cache
        cache_dir = path.join(DATA_DIR, 'dataloader/data/energy_efficiency')
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
        data_path = download(cache_dir, 'energy_efficiencty.xlsx', URL)
        self.df = pd.read_excel(data_path)
        table = self.df.to_numpy()
        train, val = train_test_split(table, test_size=self.val_split, shuffle=True)
        self.data = {
            'train': train,
            'val': val,
            'all': np.concatenate((train, val))
        }

