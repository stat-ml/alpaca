import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Dense(nn.Module):
    def __init__(self, layer_sizes, postprocessing=lambda x: x, activation=None, dropout_rate=0):
        super().__init__()

        if activation is None:
            self.activation = F.celu
        else:
            self.activation = activation

        self.layer_sizes = layer_sizes
        self.fcs = []
        for i, layer in enumerate(layer_sizes[:-1]):
            fc = nn.Linear(layer, layer_sizes[i + 1])
            setattr(self, 'fc' + str(i), fc)  # to register params
            self.fcs.append(fc)
        self.postprocessing = postprocessing

        self.dropout = nn.Dropout(dropout_rate)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

    def forward(self, x, dropout_rate=0.5, dropout_mask=None):
        out = self.activation(self.fcs[0](x))

        for layer_num, fc in enumerate(self.fcs[1:-1]):
            out = self.activation(fc(out))
            # out = self.dropout(out)

        if dropout_mask is not None:
            out = out * dropout_mask(out, dropout_rate, layer_num=0)
        else:
            out = self.dropout(out)

        out = self.fcs[-1](out)
        return out
