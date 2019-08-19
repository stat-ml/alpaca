import numpy as np
import torch
import torch.nn as nn

from dataloader.custom_dataset import loader


class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        if isinstance(x, np.ndarray):
            out = torch.DoubleTensor(x).to(self.device)
        else:
            out = x
        for fc in self.fcs:
            out = self.relu(fc(out))
            out = nn.Dropout(dropout_rate)(out)
        return out if train else out.detach()

    def fit(
            self, train_set, val_set, learning_rate=1e-4, epochs=10000,
            verbose=True, validation_step=100, patience=3, batch_size=500):
        train_loader = loader(*train_set, batch_size=batch_size)

        # optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        best_val_loss = float('inf')
        current_patience = patience

        if verbose:
            val_loss = self.evaluate(val_set)
            self._print_status(0, epochs, float('inf'), val_loss)

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

            if (epoch + 1) % validation_step == 0:
                val_loss = self.evaluate(val_set)
                if verbose:
                    self._print_status(epoch, epochs, loss.item(), val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    current_patience = patience
                else:
                    current_patience -= 1
                    if current_patience <= 0:
                        print('No patience left')
                        break

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
