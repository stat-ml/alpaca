import torch
from torch.nn.modules.loss import _Loss


class NLLRegLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', eps=1e-10):
        super(NLLRegLoss, self).__init__(size_average, reduce, reduction)
        self.eps = eps
    
    def forward(self, input, target):
        mu, sigma_sq = input[:, 0], input[:, 1] ** 2
        target = target.reshape_as(mu)
        output = torch.log(sigma_sq + self.eps)/2 + (target - mu)**2 / (2 * sigma_sq)
        return getattr(output, self.reduction)()