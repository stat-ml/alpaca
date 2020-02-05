from .mlp import MLP
from collections import OrderedDict
from itertools import count
import torch


class MLPEnsemble:
    def __init__(self, layers, n_models, reduction='mean', **kwargs):
        self.n_models = n_models
        self.models = [MLP(layers, **kwargs) for i in range(n_models)]
        self.reduction = reduction

    def fit(self, train_set, val_set, verbose=True, **kwargs):
        for i, model in enumerate(self.models): 
            if verbose:
                self._print_fit_status(i+1, self.n_models)
            model.fit(train_set, val_set, verbose=verbose, **kwargs)
    
    def state_dict(self):
        state_dict = OrderedDict({'{} model'.format(n): m.state_dict() 
                                  for n, m in zip(count(), self.models)})
        return state_dict
    
    def load_state_dict(self, state_dict):
        for n, m in enumerate(self.models):
            m.load_state_dict(state_dict['{} model'.format(n)])
    
    def train(self):
        [m.train() for m in self.models]
        
    def eval(self):
        [m.eval() for m in self.models]
    
    def __call__(self, x, reduction='default', **kwargs):
        if 'dropout_mask' in kwargs and isinstance(kwargs['dropout_mask'], list):
            masks = kwargs.pop('dropout_mask')
            res = torch.stack([m(x, dropout_mask = dpm, **kwargs) for m, dpm in zip(self.models, masks)])
        else:
            res = torch.stack([m(x, **kwargs) for m in self.models])

        if reduction == 'default':
            reduction = self.reduction

        if reduction is None:
            res = res
        elif reduction == 'mean':
            res = res.mean(dim=0)
        elif reduction == 'nll':
            means = res[:, :, 0]
            sigmas = res[:, :, 1]
            res = torch.stack([means.mean(dim=0), sigmas.mean(dim=0) +
                               (means**2).mean(dim=0) - means.mean(dim=0)**2], dim=1)
        return res
    
    def _print_fit_status(self, n_model, n_models):
        print('Fit [{}/{}] model:'.format(n_model, n_models))
