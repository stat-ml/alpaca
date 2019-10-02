import numpy as np
import torch  
    
class EnsembleMCDUE:
    """
    Estimate uncertainty for samples with Ensemble and MCDUE approach
    """ 
    def __init__(self, net, nn_runs=5, dropout_rate=.5):
        self.net = net
        self.nn_runs = nn_runs
        self.dropout_rate = dropout_rate
        self.n_models = self.net.n_models
    
    def estimate(self, X_pool, **kwargs):
        mcd_realizations = []

        with torch.no_grad():
            for nn_run in range(self.nn_runs):
                prediction = self.net(X_pool, reduction=None, dropout_rate=self.dropout_rate, **kwargs)
                prediction = prediction.to('cpu')
                mcd_realizations.append(prediction)
        
        mcd_realizations = torch.cat(mcd_realizations, dim=1)
        return np.ravel(mcd_realizations.std(dim=1, unbiased=False))
    
    
class EnsembleUE(EnsembleMCDUE):
    """
    Estimate uncertainty for samples with Ensemble approach
    """
    def __init__(self, net):
        super(EnsembleUE, self).__init__(net, nn_runs=1, dropout_rate=0)