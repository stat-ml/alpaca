import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
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


class Inferencer:
    def __init__(self, model, batch_size=8192):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, x, dropout_rate=0.5, dropout_mask=None):
        predictions = []
        self.model.eval()

        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)

        for batch in DataLoader(TensorDataset(x), batch_size=self.batch_size):
            batch = batch[0].cuda()
            prediction = self.model(batch, dropout_rate=dropout_rate, dropout_mask=dropout_mask).detach().cpu() #.numpy()

            predictions.append(prediction)

        if predictions:
            return torch.cat(predictions)
        else:
            return torch.Tensor([])

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

