import torch
import torch.nn as nn
import torch.nn.functional as F


class AnotherConv(nn.Module):
    def __init__(self, num_classes=10, activation=None):
        if activation is None:
            self.activation = F.elu
        else:
            self.activation = activation
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.linear_size = 14*14*32
        self.fc1 = nn.Linear(self.linear_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, dropout_rate=0.5, dropout_mask=None):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.linear_size)
        # x = self._dropout(x, dropout_mask, dropout_rate, 0)
        # x = self.dropout(x)
        x = self.activation(self.fc1(x))
        x = self._dropout(x, dropout_mask, dropout_rate, 1)
        x = self.fc2(x)
        return x

    def _dropout(self, x, dropout_mask, dropout_rate, layer_num):
        if dropout_mask is None:
            x = self.dropout(x)
        else:
            x = x * dropout_mask(x, dropout_rate, layer_num)
        return x
