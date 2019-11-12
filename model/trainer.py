import torch
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


    def fit(self, x, y, epochs=10, log_interval=50, verbose=False):
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

                if verbose:
                    if batch_idx % log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(loader.dataset),
                                       100. * batch_idx / len(loader), loss.item()))

    def predict(self, x):
        self.eval()
        x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            predictions = self.model(x).argmax(dim=1, keepdim=True)
        return predictions.cpu().numpy()

    def _to_loader(self, x, y, shuffle=True):
        ds = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y.reshape(-1)))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)
        return loader

    def __call__(self, x, dropout_rate=0., dropout_mask=None):
        x = torch.FloatTensor(x).to(self.device)
        return self.model(x, dropout_rate=dropout_rate, dropout_mask=dropout_mask)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class EnsembleTrainer:
    def __init__(
            self, model_class, model_kwargs, n_models, batch_size=128,
            dropout_train=0.5, reduction='mean'):
        self.models = [model_class(**model_kwargs) for _ in range(n_models)]
        self.trainers = [
            Trainer(model, batch_size=batch_size, dropout_train=dropout_train)
            for model in self.models]
        self.reduction = reduction

    def fit(self, x, y, epochs=10, log_interval=10):
        for trainer in self.trainers:
            trainer.fit(x, y, epochs, log_interval)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            predictions = self(x).argmax(dim=1, keepdim=True)
        return predictions.cpu().numpy()

    def __call__(self, x, reduction='default', **kwargs):
        res = torch.stack([trainer(x, **kwargs) for trainer in self.trainers])

        if reduction == 'default':
            reduction = self.reduction

        if reduction is None:
            res = res
        elif reduction == 'mean':
            res = res.mean(dim=0)

        return res

    def train(self):
        [trainer.train() for trainer in self.trainers]

    def eval(self):
        [trainer.eval() for trainer in self.trainers]
