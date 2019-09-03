import torch
from torch.autograd import Variable
from pyDOE import lhs
import numpy as np


class NullMask:
    def generate(self, x):
        return None


class BasicMask:
    def __init__(self, nn_runs=25):
        self.nn_runs = nn_runs

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        p = 1 - dropout_rate
        probability_tensor = x.data.new(x.data.size()[-1]).fill_(p)
        dummy_tensor = torch.bernoulli(probability_tensor)
        mask = Variable(dummy_tensor) / (p + 1e-10)
        return mask


class LHSMask:
    def __init__(self, nn_runs=25, shuffle=False):
        self.nn_runs = nn_runs
        self.layers = {}
        self.shuffle = shuffle

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        if layer_num not in self.layers:
            masks = lhs(n=x.shape[-1], samples=self.nn_runs)
            if self.shuffle:
                np.random.shuffle(masks)
            self.layers[layer_num] = iter(masks)
        mask = next(self.layers[layer_num])
        mask = (mask > dropout_rate).astype('float') / (1-dropout_rate+1e-10)
        return x.data.new(mask)

    def reset(self):
        self.layers = {}


lhs_shuffled = None
mirror_random = None
decorellating = None

