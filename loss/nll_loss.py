import torch
from torch.nn.modules.loss import _Loss


class NLLRegLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', eps=1e-6):
        super(NLLRegLoss, self).__init__(size_average, reduce, reduction)
        self.eps = eps
    
    def forward(self, input, target):
        mu, sigma = input[:, 0], input[:, 1]
        target = target.reshape_as(mu)
        output = 0.5 * torch.log(sigma + self.eps) + 0.5 * (target - mu)**2 / sigma
        return getattr(output, self.reduction)()