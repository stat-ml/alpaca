import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader


class Trainer:
    def __init__(self, model, lr=1e-3, batch_size=128, log_interval=10):
        self.model = model
        self.batch_size = batch_size
        self.device = 'cuda'
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.log_interval = log_interval


    def fit(self, x, y, epochs=10):
        self.model.train()
        loader = self._to_loader(x, y, shuffle=True)

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(loader.dataset),
                                   100. * batch_idx / len(loader), loss.item()))

    def predict(self, x):
        self.model.eval()
        x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            predictions = self.model(x).argmax(dim=1, keepdim=True)
        return predictions.cpu().numpy()

    def _to_loader(self, x, y, shuffle=False):
        ds = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y.reshape(-1)))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)
        return loader


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(config, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % config['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

