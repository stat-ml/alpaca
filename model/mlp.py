import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader.custom_dataset import loader


class MLP(nn.Module):
    def __init__(self, layer_sizes, l2_reg=1e-5):
        super(MLP, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.MSELoss()

        self.layer_sizes = layer_sizes
        self.fcs = []
        for i, layer in enumerate(layer_sizes[:-1]):
            fc = nn.Linear(layer, layer_sizes[i+1])
            setattr(self, 'fc'+str(i), fc)  # to register params
            self.fcs.append(fc)

        self.optimizer = torch.optim.Adadelta(self.parameters(), weight_decay=l2_reg)

        self.double()
        self.to(self.device)

    def forward(self, x, dropout_rate=0, train=False, dropout_mask=None):
        out = torch.DoubleTensor(x).to(self.device) if isinstance(x, np.ndarray) else x

        for layer_num, fc in enumerate(self.fcs[:-1]):
            out = F.leaky_relu(fc(out))
            if dropout_mask is None:
                out = nn.Dropout(dropout_rate)(out)
            else:
                mask = dropout_mask(out, dropout_rate, layer_num)
                out = out*mask
        out = F.leaky_relu(self.fcs[-1](out))
        return out if train else out.detach()

    def fit(
            self, train_set, val_set, epochs=10000,
            verbose=True, validation_step=100, patience=5, batch_size=500):
        train_loader = loader(*train_set, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        current_patience = patience

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
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Print intermediate results and check patience
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
        self.val_loss = val_loss

    def evaluate(self, dataset):
        """ Return model losses for provided data loader """
        data_loader = loader(*dataset)
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
