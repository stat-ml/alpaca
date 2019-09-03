from collections import defaultdict

import torch
from torch.autograd import Variable
from pyDOE import lhs
import numpy as np


class BasicMask:
    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        p = 1 - dropout_rate
        probability_tensor = x.data.new(x.data.size()[-1]).fill_(p)
        mask = torch.bernoulli(probability_tensor) / (p + 1e-10)
        return mask


class LHSMask:
    def __init__(self, nn_runs=25, shuffle=False):
        self.nn_runs = nn_runs
        self.layer_masks = {}
        self.shuffle = shuffle

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        if layer_num not in self.layer_masks:
            masks = lhs(n=x.shape[-1], samples=self.nn_runs)
            if self.shuffle:
                np.random.shuffle(masks)
            self.layer_masks[layer_num] = iter(masks)
        mask = next(self.layer_masks[layer_num])
        mask = (mask > dropout_rate).astype('float') / (1-dropout_rate+1e-10)
        return x.data.new(mask)

    def reset(self):
        self.layer_masks = {}


class MirrorMask:
    def __init__(self):
        self.layer_masks = defaultdict(list)

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        if not self.layer_masks[layer_num]:
            next_couple = self._generate_couple(x, dropout_rate)
            self.layer_masks[layer_num].extend(next_couple)
        return self.layer_masks[layer_num].pop()

    def _generate_couple(self, x, dropout_rate):
        p = 1 - dropout_rate
        probability_tensor = x.data.new(x.data.size()[-1]).fill_(p)
        mask_1 = torch.bernoulli(probability_tensor) / (p + 1e-10)
        mask_2 = x.data.new(x.data.size()[-1]).fill_(1) - mask_1 / (1 - p + 1e-10)

        return [mask_1, mask_2]


