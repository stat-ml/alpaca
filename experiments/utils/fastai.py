import torch
from torch.utils.data import Dataset
import fastai
from fastai.vision import Image


# For fastai pbar work in notebooks in vscode and pycharm
from fastprogress.fastprogress import force_console_behavior
master_bar, progress_bar = force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar


class ImageArrayDS(Dataset):
    def __init__(self, images, labels, tfms=None):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
        self.tfms = tfms

    def __getitem__(self, idx):
        image = Image(self.images[idx])
        if self.tfms is not None:
            image = image.apply_tfms(self.tfms)
        return image, self.labels[idx]

    def __len__(self):
        return len(self.images)
