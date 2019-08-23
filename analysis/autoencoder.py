import torch
import torch.nn as nn
from dataloader.custom_dataset import loader


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(Encoder, self).__init__()

        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(VAE, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.input_size = input_size
        self.encoder = Encoder(input_size, hidden_size, embedding_size)
        self.decoder = Encoder(embedding_size, hidden_size, input_size)

        l2_reg = 0
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adadelta(self.parameters(), weight_decay=l2_reg)

        self.to(self.device)
        self.double()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, x_train, x_val, epochs=100000):
        train_loader = loader(x_train, x_train)
        patience = 10
        best_val_loss = float('inf')
        current_patience = patience

        for epoch in range(epochs):
            for points, _ in train_loader:
                # Move tensors to the configured device
                points = points.reshape(-1, self.input_size).to(self.device)

                # Forward pass
                outputs = self(points)
                loss = self.criterion(outputs, points)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Print intermediate results and check patience
            if (epoch + 1) % 100 == 0:
                val_loss = self.evaluate(x_val)
                print(epoch+1, val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    current_patience = patience
                else:
                    current_patience -= 1
                    if current_patience <= 0:
                        print('No patience left')
                        break

    def predict(self, x_batch):
        data_loader = loader(x_batch, x_batch)
        result = []
        with torch.no_grad():
            for points, _ in data_loader:
                result.extend(self(points).tolist())
        return result

    def evaluate(self, x_val):
        """ Return model losses for provided data loader """
        data_loader = loader(x_val, x_val)
        with torch.no_grad():
            losses = []
            for points, _ in data_loader:
                points = points.reshape(-1, self.input_size).to(self.device)
                outputs = self(points)
                losses.append(self.criterion(outputs, points).item())

        return sum(losses)/len(losses)




