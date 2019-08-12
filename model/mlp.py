import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()

        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss()

        self.layer_sizes = layer_sizes
        self.fcs = []
        for i, layer in enumerate(layer_sizes[:-1]):
            self.fc = nn.Linear(layer, layer_sizes[i+1])  # we use self to register as torch parameter
            self.fcs.append(self.fc)
        self.relu = nn.ReLU()

    def forward(self, x, dropout=True):
        out = x
        for fc in self.fcs:
            out = self.relu(fc(out))
        return out

    def fit(self, train_loader, val_loader, learning_rate=0.001, epochs=10):
        print('start fitting')
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.reshape(-1, self.layer_sizes[0]).to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self(images, False)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    val_loss, _ = self.evaluate(val_loader)
                    self._print_status(epoch, epochs, i, loss.item(), val_loss)

    def evaluate(self, loader):
        """
        Return model losses and accuracy for provided data loader
        """
        with torch.no_grad():
            correct = 0
            total = 0
            losses = []
            for images, labels in loader:
                images = images.reshape(-1, self.layer_sizes[0]).to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                losses.append(self.criterion(outputs, labels).item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return sum(losses)/len(losses), correct/total



    def _print_status(self, epoch, epochs, i, loss, val_loss):
        print('Epoch [{}/{}], Step [{}], Loss: {:.4f}, Validation loss: {:.4f}'
              .format(epoch + 1, epochs, i + 1, loss, val_loss))
