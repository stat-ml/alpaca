import os
from pathlib import Path

import tarfile

from experiments.experiment_setup import ROOT_DIR
from .saver import DataSaver
from .downloader import download


URL = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz'


class NotMNISTData:
    def __init__(self, use_cache=False, val_split=0.2):
        raise NotImplementedError

        self.use_cache = use_cache
        cache_dir = Path(ROOT_DIR)/'dataloader'/'data'/'not_mnist'
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
        data_path = download(cache_dir, 'notMNIST_small.tar.gz', URL)
        import ipdb; ipdb.set_trace()
        with tarfile.open(data_path, "r:gz") as tar_ref:
            tar_ref.extractall(cache_dir)

        base_dir = cache_dir / 'notMNIST_small'

        class_dirs = sorted(os.listdir(base_dir))
        # with zipfile.ZipFile(data_path, 'r') as zip_ref:
        #     zip_ref.extractall(cache_dir)

        # train, val = train_test_split(table, test_size=self.val_split, shuffle=True)
        # self.data = {
        #     'train': train,
        #     'val': val,
        #     'all': np.concatenate((train, val))
        # }


if __name__ == '__main__':
    dataset = NotMNISTData()
    x_train, y_train = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')
    print(x_train.shape, y_train.shape, y_val.shape)
    print(x_train[:5])
    print(y_train[:5])