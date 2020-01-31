import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class Trainer:
    def __init__(
            self, model, batch_size=128, lr=1e-3, dropout_train=0.5, weight_decay=1e-4,
            loss=None, regression=False):
        self.model = model
        self.device = 'cuda'
        self.model.to(self.device)

        # self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=weight_decay)
        self.dropout_train = dropout_train
        self.batch_size = batch_size

        self.loss = loss or F.cross_entropy
        self.regression = regression

        self.val_loss_history = []
        self.train_loss_history = []

    def fit(self, train_set, val_set, epochs=10, log_interval=1000, verbose=False, patience=5, dropout_rate=None):
        if dropout_rate is None:
            dropout_rate = self.dropout_train

        self.model.train()
        loader = self._to_loader(*train_set)
        val_loader = self._to_loader(*val_set)

        self._set_patience(patience)

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data, dropout_rate=dropout_rate)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()

                if verbose:
                    self._report(epoch, loss, data, batch_idx, log_interval, loader, val_loader)

            if not self._check_patience(val_loader):
                break

    def _set_patience(self, patience):
        self.start_patience = patience
        self.current_patience = self.start_patience
        self.best_loss = float('inf')

    def _check_patience(self, val_loader):
        loss = self.evaluate(val_loader)
        if loss < self.best_loss:
            self.best_loss = loss
            self.current_patience = self.start_patience
        else:
            self.current_patience -= 1

        return self.current_patience > 0

    def _report(self, epoch, loss, data, batch_idx, log_interval, loader, val_loader):
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(loader) - 1:
            val_loss = self.evaluate(val_loader)
            percent = batch_idx / len(loader)
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tVal Loss: {:.6f}".format(
                epoch, int(percent*len(loader.dataset)), len(loader.dataset), 100 * percent,
                loss.item(), val_loss))

            self.val_loss_history.append(val_loss)
            self.train_loss_history.append(loss)

    def evaluate(self, val_loader):
        if not isinstance(val_loader, DataLoader):
            if len(val_loader) == 1:
                val_loader = self._to_loader(val_loader)
            elif len(val_loader) == 2:
                val_loader = self._to_loader(val_loader[0], val_loader[1])

        losses = []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data, dropout_rate=0)
                losses.append(self.loss(output, target).item())
        return np.mean(losses)

    def predict(self, x, logits=False):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            loader = self._to_loader(x, shuffle=False)
            for batch in loader:
                batch = batch[0].to(self.device)
                prediction = self.model(batch)
                if not logits and not self.regression:
                    prediction = prediction.argmax(dim=1, keepdim=True)
                predictions.append(prediction.cpu())
            predictions = torch.cat(predictions).numpy()
        return predictions

    def _to_loader(self, x, y=None, shuffle=True):
        if y is None:
            ds = TensorDataset(torch.FloatTensor(x))
        else:
            if self.regression:
                ds = TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y))
            else:
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
        self.device = 'cuda'

    def fit(self, train_set, val_set, **kwargs):
        for trainer in self.trainers:
            trainer.fit(train_set, val_set, **kwargs)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            # predictions = self(x).detach().argmax(dim=1, keepdim=True)
            predictions = self(x).argmax(axis=1)[..., np.newaxis]
        return predictions.cpu().numpy()

    def evaluate(self, val_loader):
        return np.mean([trainer.evaluate(val_loader) for trainer in self.trainers])

    def __call__(self, x, reduction='default', **kwargs):
        res = torch.stack([torch.Tensor(trainer.predict(x, logits=True)) for trainer in self.trainers])

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
