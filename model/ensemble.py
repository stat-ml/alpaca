########################################################################
#                            for debugging                             #
########################################################################
import numpy as np

class BaseModel:
    
    def __init__(self):
        pass
    
    def __call__(self, x):
        return [0] * len(x)
    
    def fit(self, train_set, val_set, **kwargs):
        pass

class RandomModel(BaseModel):
    
    def __call__(self, x):
        return np.random.rand(len(x)) + np.array(x)

########################################################################
#                                tools                                 #
########################################################################

def clone(obj, new_params=tuple()):
    """Constructs a new estimator with the same parameters.
    And update new parameters if it's necessary
    Parameters
    ----------
    obj : object
        The model to be cloned
    new_params : tuple, optional
        The new parameters of obj
    """
    klass = obj.__class__
    new_obj = klass()
    ####################
    #       TO DO      #
    #------------------#
    # ADD: copy params #
    ####################
    return new_obj

########################################################################
#                             for debuging                             #
########################################################################
class BaseEnsamble:
    def __init__(self, base_model, n_models, model_params=tuple()):
        
        self.base_model = base_model
        self.n_models = n_models
        self.model_params = model_params
        self.models_ = [clone(base_model) for i in range(n_models)]

    def fit(self, train_set, val_set, verbose=True, **kwargs):
        
        for i, model in enumerate(self.models_): 
            if verbose:
                self._print_fit_status(i, self.n_models)
            model.fit(train_set, val_set, verbose=True, **kwargs)

    def __call__(self, dataset):
        all_res = [m(dataset) for m in self.models_]
        agg_res = [self.agg_function(res) for res in zip(*all_res)]
        return agg_res

    def _print_fit_status(self, n_model, n_models):
        print('Fit [{}/{}] model:'.format(n_model, n_models))

class RegressionEnsamble(BaseEnsamble):
    
    @staticmethod
    def agg_function(arr):
        return sum(arr)/len(arr)
    
class ClassificationEnsamble(BaseEnsamble)
    
    @staticmethod
    def agg_function(arr):
        cnt = Counter(arr)
        res = cnt.most_common()[0][0]
        #################################
        #            TO DO              #
        #-------------------------------#
        # CHECK: the most common is one #
        # ADD: random choise from       #
        #      most commons             #
        #################################
        return res