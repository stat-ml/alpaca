from .mlp import MLP
from collections import OrderedDict
from itertools import count
import torch

class MLPEnsemble:
    def __init__(self, layers, n_models, **kwargs):
        
        self.n_models = n_models
        self.models = [MLP(layers, **kwargs) for i in range(n_models)]

    def fit(self, train_set, val_set, verbose=True, **kwargs):
        for i, model in enumerate(self.models): 
            if verbose:
                self._print_fit_status(i, self.n_models)
            model.fit(train_set, val_set, verbose=True, **kwargs)
    
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
    
    def __call__(self, x):
        res = torch.cat([m(x) for m in self.models], dim=1).mean(dim=1, keepdim=True)
        return res

    def _print_fit_status(self, n_model, n_models):
        print('Fit [{}/{}] model:'.format(n_model, n_models))