from os import path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import zipfile

from .config import DATA_DIR
from .saver import DataSaver
from .downloader import download

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip"


class CTSlicesData:
    """Relative location of CT slices on axial axis, UCI dataset"""

    def __init__(self, use_cache=False, val_split=0.2):
        self.use_cache = use_cache
        cache_dir = path.join(DATA_DIR, "dataloader/data/ct_slices")
        self.saver = DataSaver(cache_dir)
        self.val_split = val_split
        self._build_dataset(cache_dir)

    def dataset(self, label):
        if self.use_cache:
            return self.saver.load(label)

        data = self.data[label]
        x, y = data[:, :-1], data[:, -1:]
        self.saver.save(x, y, label)
        return x, y

    def _build_dataset(self, cache_dir):
        data_path = download(cache_dir, "slice_localization_data.zip", URL)
        with zipfile.ZipFile(data_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)
        file_path = path.join(cache_dir, "slice_localization_data.csv")
        self.df = pd.read_csv(file_path)
        self.df.drop(['patientId'],
                     axis=1,
                     inplace=True)

        table = self.df.to_numpy()

        if self.val_split != 0:
            train, val = train_test_split(table, test_size=self.val_split, shuffle=True)
        else:
            train, val = table, []
        self.data = {"train": train, "val": val}


if __name__ == "__main__":
    dataset = CTSlicesData()
    x_train, y_train = dataset.dataset("train")
    x_val, y_val = dataset.dataset("val")
    print(x_train.shape, y_val.shape)
