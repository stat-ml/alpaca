from os import path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR
from .saver import DataSaver
from .downloader import download


URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv'


class ProteinStructureData:
    """Protein structure (CASP) dataset from UCI"""
    def __init__(self, use_cache=False, val_split=0.2):
        self.use_cache = use_cache
        cache_dir = path.join(DATA_DIR, 'dataloader/data/protein_structure')
        self.saver = DataSaver(cache_dir)
        self.val_split = val_split
        self._build_dataset(cache_dir)

    def dataset(self, label):
        if self.use_cache:
            return self.saver.load(label)

        data = self.data[label]
        x, y = data[:, 1:], data[:, :1]
        self.saver.save(x, y, label)
        return x, y

    def _build_dataset(self, cache_dir):
        data_path = download(cache_dir, 'CASP.csv', URL)
        self.df = pd.read_csv(data_path)
        table = self.df.to_numpy()
        train, val = train_test_split(table, test_size=self.val_split, shuffle=True)
        self.data = {
            'train': train,
            'val': val,
        }


if __name__ == '__main__':
    dataset = ProteinStructureData()
    x_train, y_train = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')
    print(dataset.df.head())
    print(x_train.shape, y_train.shape, y_val.shape)

