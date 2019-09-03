import os
import os.path as path
import zipfile
import pandas as pd
import wget


class BlogFeedbackData:
    def __init__(
            self, cache_dir='dataloader/data/blog_feedback',
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip'):
        self.cache_dir = cache_dir
        self.url = url
        self.filename = 'bf.zip'

    def dataset(self, label='train'):
        try:
            return self._parse(label)
        except FileNotFoundError:
            self._load()
            self._parse(label)

    def _parse(self, label):
        if label == 'train':
            train_file = 'blogData_train.csv'
            df = pd.read_csv(os.path.join(self.cache_dir, train_file), header=None)
        elif label == 'test':
            files = os.listdir(self.cache_dir)
            test_files = [file for file in files if file.startswith('blogData_test')]
            test_files = [os.path.join(self.cache_dir, file) for file in test_files]
            df = pd.concat((pd.read_csv(f, header=None) for f in test_files))
        else:
            raise RuntimeError("Wrong label")

        return df.loc[:, :279].values, df.loc[:, 280].values

    def _load(self):
        if not path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        zip_path = path.join(self.cache_dir, self.filename)
        if not path.exists(zip_path):
            wget.download(self.url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)


if __name__ == '__main__':
    print(BlogFeedbackData().dataset())
