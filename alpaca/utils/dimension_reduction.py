import abc
from typing import Optional, List
import torch
import torch.nn as nn

# TODO: should be removed
import numpy as np
from sklearn.decomposition import PCA

__all__ = ["DRAutoencoder"]


class DR(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduce the dimension of the given tensor
        """
        ...


class DRAutoencoder(nn.Module, DR):
    """
    The basic Autoencoder implementation

    Parameters
    ----------
    input_dim : int
        The input dimension
    output_dim: int
        The dimension of vector after reduction
    hidden_dim : Optional[int] = None
        The hidden dimension
    lr : float = 1e-3:
        The learning rate for optimizer (Adam is used by default)
    epochs : int = 10
        The number of epocsh for fit function
    enc_modulelist: Optional[List[nn.Module]] = None
        User provided encoder module
    dec_modulelist: Optional[List[nn.Module]] = None
        User provided decoder module

    Example
    -------
    Using default modules:
    >>> x = next(train_loader)
    >>> reducer = DRAutoencoder(input_dim=x.size(-1), output_dim=10, lr=1e-3)
    >>> reducer.fit(train_loader)
    >>> x_reduced = reducer.reduce(x)
    Using user custom modules
    >>> output_dim = 10
    >>> encoder = nn.Sequential([\
    >>>                 nn.Linear(x.size(-1), x.size(-1)),
    >>>                 nn.Linear(x.size(-1), output_dim)
    >>>         ])
    >>> reducer = DRAutoencoder(input_dim=x.size(-1, output_dim=output_dim, lr=1e-3, enc_modulelist=encoder)
    >>> x_reduced = reducer.reduce(x)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        lr: float = 1e-3,
        epochs: int = 10,
        enc_modulelist: Optional[List[nn.Module]] = None,
        dec_modulelist: Optional[List[nn.Module]] = None,
    ):
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        if enc_modulelist is None:
            self.encoder = nn.ModuleList(
                [nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)]
            )
        else:
            self.encoder = enc_modulelist

        if dec_modulelist is None:
            self.decoder = nn.ModuleList(
                [nn.Linear(output_dim, hidden_dim), nn.Linear(hidden_dim, input_dim)]
            )
        else:
            self.decoder = dec_modulelist
        self.lr = lr
        self.epochs = epochs

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        for item in self.enc_modulelist[:-1]:
            x = torch.nn.functional.relu(item(x))
        return self.enc_modulelist[-1](x)

    def decode(self, z):
        for item in self.dec_modulelist[:-1]:
            z = torch.nn.functional.relu(item(z))
        return self.dec_modulelist[-1](z)

    def forward(self, x):
        z = self.reduce(x)
        out = self.decode(z)
        return out

    def fit(self, train_loader):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        for epoch in self.epochs:
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                recon_batch = self(data)
                loss = torch.nn.functional.mse_loss(recon_batch, data)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def evaluate(self, eval_loader):
        self.eval()
        eval_loss = 0.0
        for batch_idx, (data, _) in enumerate(eval_loader):
            data = data.to(self.device)
            recon_batch = self(data)
            loss = torch.nn.functional.mse_loss(recon_batch, data)
            loss.backward()
            eval_loss += loss.item()
        return eval_loss / len(eval_loader.dataset)


class DRPCA(nn.Module):
    """
    The basic PCA implementation based on sklearn

    Parameters
    ----------
    output_dim: int
        The dimension of vector after reduction
    fit_n: int
        The sampled indices for reduction model to be fit on

    Example
    -------
    >>> X = X_large # large tensor
    >>> x = next(train_loader)
    >>> reducer = DRPCA(output_dim=10, fit_n=500)
    >>> reducer.fit(X)
    >>> x_reduced = reducer.reduce(x)
    """

    def __init__(self, output_dim: int, fit_n: int = 1000):
        self.output_dim = output_dim
        self.fit_n = fit_n

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """
        The implementation is based on sklearn
        """
        x = x.numpy()
        # FIXME: we need to get rid of sklearn
        x_reduced = self.pca.transform(x)
        return x_reduced

    def fit(self, X: torch.Tensor) -> torch.Tensor:
        X = X.numpy()
        inds = np.random.choice(X.size(0), min(X.size(0), self.fit_n))
        X_ = X[inds]
        self.pca = PCA(n_components=self.output_dim)
        self.pca.fit(X_)
