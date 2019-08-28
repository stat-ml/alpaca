import torch
from torch.autograd import Variable


def basic_mask(x, dropout_rate=0.5):
    p = 1 - dropout_rate
    dummy_tensor = torch.bernoulli(x.data.new(x.data.size()).fill_(p))
    mask = Variable(dummy_tensor) / (p + 1e-10)
    return mask
