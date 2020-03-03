from .sample_selector import EagerSampleSelector
from .oracle import IdentityOracle


class ALTrainer:
    """
    Active Learning trainer

    trains on train data
    on each iteration extends training sets from sampling the pool
    """
    def __init__(
            self, model, estimator, oracle=None, sampler=None, y_pool=None, iterations=10,
            update_size=100, verbose=True, patience=10, val_on_pool=False):
        self.model = model
        self.estimator = estimator
        self.sampler = sampler
        self.oracle = oracle
        self.iterations = iterations
        self.update_size = update_size
        self.verbose = verbose
        self.patience = patience
        self.val_on_pool = val_on_pool

        if sampler is None:
            sampler = EagerSampleSelector()
        self.sampler = sampler

        if oracle is None:
            if y_pool is not None:
                oracle = IdentityOracle(y_pool)
            else:
                raise ValueError("Either oracle or y_pool should be initialized")
        self.oracle = oracle

    def train(self, x_train, y_train, x_val, y_val, x_pool):
        errors = []

        for al_iteration in range(self.iterations):
            print("Iteration", al_iteration+1)
            print(x_pool.shape)
            # retrain net
            self.model.fit((x_train, y_train), (x_val, y_val), verbose=self.verbose, patience=self.patience)
            if self.val_on_pool:
                error = self.model.evaluate((x_pool, self.oracle.y_set))
            else:
                error = self.model.evaluate((x_val, y_val))

            print('Validation error after training: %.3f' % error)
            errors.append(error)

            # update pool
            if hasattr(self.estimator, 'reset'):
                self.estimator.reset()
            uncertainties = self.estimator.estimate(x_pool, x_train)
            x_train, y_train, x_pool = self.sampler.update_sets(
                x_train, y_train, x_pool, uncertainties, self.update_size, self.oracle
            )

            if self.verbose:
                print('Uncertainties', uncertainties[:20])
                print('Top uncertainties', uncertainties[uncertainties.argsort()[-10:][::-1]])

        return errors

