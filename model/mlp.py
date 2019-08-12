import numpy as np
import torch
import torch.nn as nn

from dataloader.custom_dataset import loader


class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()

        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.MSELoss()

        self.layer_sizes = layer_sizes
        self.fcs = []
        for i, layer in enumerate(layer_sizes[:-1]):
            fc = nn.Linear(layer, layer_sizes[i+1])
            setattr(self, 'fc'+str(i), fc)  # to register params
            self.fcs.append(fc)
        self.relu = nn.ReLU()

        self.double()
        self.to(self.device)

    def forward(self, x, dropout_rate=0, train=False):
        out = torch.DoubleTensor(x)
        for fc in self.fcs:
            out = self.relu(fc(out))
            out = nn.Dropout(dropout_rate)(out)
        return out if train else out.detach()

    def fit(self, train_set, val_set, learning_rate=0.001, epochs=1000, verbose=True):
        train_loader = loader(*train_set)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(epochs):
            for points, labels in train_loader:
                # Move tensors to the configured device
                points = points.reshape(-1, self.layer_sizes[0]).to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self(points, train=True)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 50 == 0 and verbose:
                val_loss = self.evaluate(val_set)
                self._print_status(epoch, epochs, loss.item(), val_loss)

    def evaluate(self, dataset):
        data_loader = loader(*dataset)
        """
        Return model losses for provided data loader
        """
        with torch.no_grad():
            losses = []
            for points, labels in data_loader:
                points = points.reshape(-1, self.layer_sizes[0]).to(self.device)
                labels = labels.to(self.device)
                outputs = self(points)
                losses.append(self.criterion(outputs, labels).item())

        return sum(losses)/len(losses)

    def _print_status(self, epoch, epochs, loss, val_loss):
        print('Epoch [{}/{}], Loss: {:.4f}, Validation loss: {:.4f}'
              .format(epoch + 1, epochs, loss, val_loss))
