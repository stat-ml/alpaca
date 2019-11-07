import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader


class Trainer:
    def __init__(self, model, batch_size=128, lr=1e-3, dropout_train=0.5):
        self.model = model
        self.device = 'cuda'
        self.model.to(self.device)

        # self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.optimizer = torch.optim.Adadelta(model.parameters())
        self.dropout_train = dropout_train
        self.batch_size = batch_size


    def fit(self, x, y, epochs=10, log_interval=50):
        self.model.train()
        loader = self._to_loader(x, y)

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data, dropout_rate=0.25)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(loader.dataset),
                                   100. * batch_idx / len(loader), loss.item()))

    def predict(self, x):
        self.model.eval()
        x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            predictions = self.model(x).argmax(dim=1, keepdim=True)
        return predictions.cpu().numpy()

    def _to_loader(self, x, y, shuffle=True):
        ds = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y.reshape(-1)))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)
        return loader

    def __call__(self, x, dropout_rate=0.5, dropout_mask=None):
        self.model.train()
        x = torch.FloatTensor(x).to(self.device)
        return self.model(x, dropout_rate=dropout_rate, dropout_mask=dropout_mask)


class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(12*12*64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, dropout_rate=0., dropout_mask=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 12*12*64)
        print(x.shape)
        x = self._dropout(x, dropout_mask, dropout_rate, 0)
        x = F.relu(self.fc1(x))
        x = self._dropout(x, dropout_mask, dropout_rate, 1)
        x = self.fc2(x)
        return x

    def _dropout(self, x, dropout_mask, dropout_rate, layer_num):
        if dropout_mask is None:
            x = self.dropout(x)
        else:
            x = x * dropout_mask(x, dropout_rate, layer_num)
        return x


