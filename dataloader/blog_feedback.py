import os
import os.path as path
import zipfile
import wget


class BlogFeedbackData:
    def __init__(
            self, cache_dir='dataloader/data/blog_feedback',
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip'):
        self.cache_dir = cache_dir
        self.url = url
        self.filename = 'bf.zip'

    def dataset(self):
        try:
            return self._parse()
        except Exception:
            self._load()
            self._parse()

    def _parse(self):
        return []

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
