import torch
import torch.nn as nn
from torch.nn import functional as F
from dataloader.custom_dataset import loader


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(VAE, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, embedding_size)
        self.fc22 = nn.Linear(hidden_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.double()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def _loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_size), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def fit(self, x_train):
        self.train()
        train_loss = 0
        train_loader = loader(x_train, x_train)
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self(data)
            loss = self._loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

    def evaluate(self, x_val):
        self.eval()
        test_loss = 0
        test_loader = loader(x_val, x_val)
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self(data)
                test_loss += self._loss_function(recon_batch, data, mu, logvar).item()

        test_loss /= len(test_loader.dataset)
        return test_loss

    def predict(self, x_batch):
        data_loader = loader(x_batch, x_batch)
        result = []
        with torch.no_grad():
            for points, _ in data_loader:
                result.extend(self(points)[0].tolist())
        return result

