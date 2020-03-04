import os.path as path

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from .config import DATA_DIR
from .downloader import download

URL = 'http://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/in_situ_co2/monthly/monthly_in_situ_co2_mlo.csv'


class CO2Data:
    def __init__(self, val_split=0.2):
        self.cache_dir = path.join(DATA_DIR, 'dataloader/data/co_2', )
        self.val_split = val_split
        self._build_dataset(self.cache_dir)

    def dataset(self, label='train'):
        data = self.data[label]
        x, y = data[:, :1], data[:, 1:]
        return x, y

    def _build_dataset(self, cache_dir):
        data_path = download(cache_dir, 'co2_monthly.csv', URL)
        df = pd.read_csv(data_path, comment='"').drop([0, 1]).iloc[:, [3, 4]]
        for column in df:
            df[column] = df[column].astype(np.float)
        df = df[df.iloc[:, 1] != -99.99]

        table = df.to_numpy()
        if self.val_split == 0:
            train = table
            val = None
        else:
            train, val = train_test_split(table, test_size=self.val_split)

        self.data = {'train': train, 'val': val}


if __name__ == '__main__':
    CO2Data()
