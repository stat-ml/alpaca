class ALTrainer:
    """
    Active Learning trainer

    trains on train data
    on each iteration extends training sets from sampling the pool
    """
    def __init__(
            self, model, estimator, sampler, oracle, iterations=10,
            update_size=100, verbose=True, patience=10):
        self.model = model
        self.estimator = estimator
        self.sampler = sampler
        self.oracle = oracle
        self.iterations = iterations
        self.update_size = update_size
        self.verbose = verbose
        self.patience = patience

    def train(self, x_train, y_train, x_val, y_val, x_pool):
        rmses = []

        for al_iteration in range(self.iterations):
            print("Iteration", al_iteration+1)
            # retrain net
            self.model.fit((x_train, y_train), (x_val, y_val), verbose=self.verbose, patience=self.patience)
            rmse = self.model.evaluate((x_val, y_val))
            print('Validation RMSE after training: %.3f' % rmse)
            rmses.append(rmse)

            # update pool
            uncertainties = self.estimator.estimate(x_pool, x_train)
            x_train, y_train, x_pool = self.sampler.update_sets(
                x_train, y_train, x_pool, uncertainties, self.update_size, self.oracle
            )

            if self.verbose:
                print('Uncertainties', uncertainties[:20])
                print('Top uncertainties', uncertainties[uncertainties.argsort()[-10:][::-1]])

        return rmses

