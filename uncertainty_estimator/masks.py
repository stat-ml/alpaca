import torch
from torch.autograd import Variable


class NullMask:
    def generate(self, x):
        return None


class BasicMask:
    def __init__(self, nn_runs=25):
        self.nn_runs = nn_runs

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        p = 1 - dropout_rate
        dummy_tensor = torch.bernoulli(x.data.new(x.data.size()).fill_(p))
        mask = Variable(dummy_tensor) / (p + 1e-10)
        return mask

    # def generate(self, x, dropout_rate=0.5):
    #     p = 1 - dropout_rate
    #     dummy_tensor = torch.bernoulli(x.data.new(x.data.size()).fill_(p))
    #     mask = Variable(dummy_tensor) / (p + 1e-10)
    #     return mask


class LHSMask:
    def __init__(self, nn_runs=25):
        self.nn_runs = nn_runs

    def generate(self, x, dropout_rate=0.5):
        return x.data.new(x.data.size()).fill_(1)


lhs_shuffled = None
mirror_random = None
decorellating = None

