import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from alpaca.utils.datasets.builder import build_dataset
from alpaca.ue import masks
import alpaca.nn as ann


class SimpleConv(nn.Module):
    def __init__(self, num_classes=10, activation=None, dropout_rate=0.5):
        if activation is None:
            self.activation = F.leaky_relu
        else:
            self.activation = activation
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.linear_size = 12 * 12 * 32
        self.fc1 = nn.Linear(self.linear_size, 256)
        self.dropout = ann.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, dropout_mask=None):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.linear_size)
        x = self.activation(self.fc1(x))
        x = self.dropout(x, dropout_mask=dropout_mask, layer_num=0)
        x = self.fc2(x)
        return x


@pytest.fixture(scope="function", autouse=True)
def seed():
    torch.manual_seed(123)
    yield


@pytest.fixture(
    scope="function",
    params=[
        "mc_dropout",
        "decorelating",
        "decorelating_sc",
        "ht_decorrelating",
        "leveragescoremask",
        "ht_leverages",
        "cov_leverages",
    ],
)
def mask(request):
    mask_name = request.param
    return masks.reg_masks[mask_name]


@pytest.fixture(scope="function", params=[("mnist", 10000)])
def dataset(request):
    dataset, val_size = request.param
    dataset = build_dataset(dataset, val_size=val_size)
    x_train, y_train = dataset.dataset("train")
    x_val, y_val = dataset.dataset("val")
    x_shape = dataset.x_shape

    train_ds = TensorDataset(
        torch.FloatTensor(x_train.reshape(x_shape)), torch.LongTensor(y_train)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(x_val.reshape(x_shape)), torch.LongTensor(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=512)
    val_loader = DataLoader(val_ds, batch_size=512)
    return train_loader, val_loader


@pytest.fixture(scope="module")
def simple_conv(dataset):
    train_loader, val_loader = dataset
    model = SimpleConv()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for x_batch, y_batch in train_loader:  # Train for one epoch
        prediction = model(x_batch)
        optimizer.zero_grad()
        loss = criterion(prediction, y_batch)
        loss.backward()
        optimizer.step()

    x_batch, y_batch = next(iter(val_loader))
    return model, x_batch
