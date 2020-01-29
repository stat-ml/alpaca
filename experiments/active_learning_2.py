import sys
sys.path.append('..')

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from fastai.vision import rand_pad, flip_lr, ImageDataBunch, Learner, accuracy, Image

from model.model_alternative import AnotherConv
from dataloader.builder import build_dataset
from uncertainty_estimator.bald import Bald, BaldMasked
from uncertainty_estimator.masks import build_masks, build_mask


# plt.switch_backend('Qt4Agg')  # to work with remote server
# torch.cuda.set_device(1)
torch.backends.cudnn.benchmark = True


total_size = 60_000
val_size = 10_000
start_size = 5_000
step_size = 500
steps = 20
retrain = False


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

    def get_state(self):
        pass


# Load data
dataset = build_dataset('cifar_10', val_size=10_000)
x_set, y_set = dataset.dataset('train')
x_val, y_val = dataset.dataset('val')

shape = (-1, 3, 32, 32)
x_set = ((x_set - 128)/128).reshape(shape)
x_val = ((x_val - 128)/128).reshape(shape)

x_pool, x_train, y_pool, y_train = train_test_split(x_set, y_set, test_size=start_size, stratify=y_set)

train_tfms = [*rand_pad(4, 32), flip_lr(p=0.5)]
train_ds = ImageArrayDS(x_train, y_train, train_tfms)
val_ds = ImageArrayDS(x_val, y_val)
data = ImageDataBunch.create(train_ds, val_ds, bs=256)


loss_func = torch.nn.CrossEntropyLoss()
model = AnotherConv()
learner = Learner(data, model, metrics=accuracy, loss_func=loss_func)

model_path = "experiments/data/model.pt"
if retrain:
    learner.fit(2, 3e-3, wd=0.2)
    torch.save(model.state_dict(), model_path)
else:
    model.load_state_dict(torch.load(model_path))


images = torch.FloatTensor(x_val[:50]).to('cuda')
mask = build_mask('rank_l_dpp')
estimator = BaldMasked(model, dropout_mask=mask, num_classes=10)
estimations = estimator.estimate(images)
idxs = np.argsort(estimations)[::-1]
print(idxs)
print(estimations[idxs])

