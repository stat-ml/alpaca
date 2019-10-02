import numpy as np
import torch  

class EnsembleUE:
    """
    Estimate uncertainty for samples with Ensemble approach
    """
    def __init__(self, net):
        self.net = net
        
    def estimate(self, X_pool, *args):

        with torch.no_grad():
            predictions = self.net(X_pool, reduction=None).to('cpu')

        return np.ravel(predictions.std(dim=0))
    
class EnsembleMCDUE:
    """
    Estimate uncertainty for samples with Ensemble and MCDUE approach
    """ 
    def __init__(self, net, nn_runs=5, dropout_rate=.5):
        self.net = net
        self.nn_runs = nn_runs
        self.dropout_rate = dropout_rate
        self.n_models = self.net.n_models
    
    def estimate(self, X_pool, *args):
        mcd_realizations = []

        with torch.no_grad():
            for nn_run in range(self.nn_runs):
                prediction = self.net(X_pool, reduction=None, dropout_rate=self.dropout_rate)
                prediction = np.ravel(prediction.to('cpu')).reshape((X_pool.shape[0], self.n_models))
                mcd_realizations.append(prediction)

        mcd_realizations = np.concatenate(mcd_realizations, axis=1)
        return np.ravel(np.std(mcd_realizations, axis=1))