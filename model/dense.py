import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Dense(nn.Module):
    def __init__(self, layer_sizes, postprocessing=lambda x: x, activation=None):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if activation is None:
            self.activation = F.elu
        else:
            self.activation = activation

        self.layer_sizes = layer_sizes
        self.fcs = []
        for i, layer in enumerate(layer_sizes[:-1]):
            fc = nn.Linear(layer, layer_sizes[i + 1])
            setattr(self, 'fc' + str(i), fc)  # to register params
            self.fcs.append(fc)
        self.postprocessing = postprocessing

        # self.double()
        self.to(self.device)

    def forward(self, x, dropout_rate=0, dropout_mask=None):
        # out = torch.FloatTensor(x).to(self.device) if isinstance(x, np.ndarray) else x
        out = x
        out = self.activation(self.fcs[0](out))

        for layer_num, fc in enumerate(self.fcs[1:-1]):
            out = self.activation(fc(out))
            if dropout_mask is None:
                out = nn.Dropout(dropout_rate)(out)
            else:
                out = out*dropout_mask(out, dropout_rate, layer_num)
        out = self.fcs[-1](out)
        return out
