from collections import OrderedDict

import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback
from torch.utils.data import TensorDataset, DataLoader
import torch

from dataloader.builder import build_dataset
from model.model_alternative import AnotherConv
from model.trainer import Trainer

matplotlib.use('tkagg')

dataset = build_dataset('cifar_10', val_size=10_000)

x_train, y_train = dataset.dataset('train')
x_val, y_val = dataset.dataset('val')

train_ds = TensorDataset(torch.FloatTensor(x_train.reshape(-1, 3, 32, 32)), torch.LongTensor(y_train.reshape(-1)))
val_ds = TensorDataset(torch.FloatTensor(x_val.reshape(-1, 3, 32, 32)), torch.LongTensor(y_val.reshape(-1)))

batch = 120
train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2)

data = OrderedDict()
data['train'] = train_dl
data['valid'] = val_dl

model = AnotherConv()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = torch.nn.CrossEntropyLoss()

runner = SupervisedRunner()
runner.train(
    model=model,
    criterion=crit,
    optimizer=optimizer,
    loaders=data,
    logdir="run",
    callbacks=[
        EarlyStoppingCallback(patience=2, min_delta=0.01)
    ],
    num_epochs=20)


print(model)

