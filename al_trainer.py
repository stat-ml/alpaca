import numpy as np
from sklearn.metrics import mean_squared_error as mse


class ALTrainer:
    """
    Active Learning trainer

    trains on train data
    on each iteration extends training sets from sampling the pool
    """
    def __init__(self, model, estimator, sampler, oracle, iterations=10, update_size=100, verbose=True):
        self.model = model
        self.estimator = estimator
        self.sampler = sampler
        self.oracle = oracle
        self.iterations = iterations
        self.update_size = update_size
        self.verbose = verbose

    def train(self, X_train, y_train, X_val, y_val, X_pool):
        self.model.train(X_train, y_train, X_val, y_val, verbose=self.verbose)

        rmses = [self._rmse(X_val, y_val)]

        for al_iteration in range(1, self.iterations + 1):
            # update pool
            uncertainties = self.estimator.estimate(X_pool, X_train, y_train)
            X_train, y_train, X_pool = self.sampler.update_sets(
                X_train, y_train, X_pool, uncertainties, self.update_size, self.oracle
            )
            if self.verbose:
                print('Uncertainties', uncertainties[:20])
                print('Top uncertainties', uncertainties[uncertainties.argsort()[-10:][::-1]])
            print("Iteration", al_iteration)

            # retrain net
            self.model.train(X_train, y_train, X_val, y_val, verbose=self.verbose)
            rmse = self._rmse(X_val, y_val)
            print('Validation RMSE after training: %.3f' % rmse)
            rmses.append(rmse)

        return rmses

    def _rmse(self, X, y):
        return np.sqrt(mse(self.model.predict(data=X), y))
