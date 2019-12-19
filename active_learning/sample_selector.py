import numpy as np


class EagerSampleSelector:
    """
    Move samples from pool dataset to train dataset
    Takes samples with highest uncertainty first
    """
    def update_sets(self, X_train, y_train, X_pool, ue_values, sample_size, oracle):
        """
        Update X_train and X_pool by choosing points with highest uncertainty estimations.
        :param X_train, y_train: the datasets that should be extended
        :param ue_values: uncertainty estimation for points in x_pool
        :param sample_size: how much points to add
        :param oracle: the class that should generate y_values for new samples
        :return:
        """
        # obtain new values from pool
        indices = self.sample(ue_values, sample_size)
        X_selected = X_pool[indices]
        Y_selected = oracle.evaluate(X_pool, indices)

        # change sets
        X_train = np.concatenate([X_train, X_selected])
        y_train = np.concatenate([y_train, Y_selected])
        X_pool = np.delete(X_pool, indices, axis=0)

        return X_train, y_train, X_pool

    @staticmethod
    def sample(uncertainty, sample_size):
        return np.argsort(uncertainty)[::-1][:sample_size]


class StochasticSampleSelector(EagerSampleSelector):
        """
        Move samples from pool dataset to train dataset
        Takes samples with probability propotional to uncertainty
        """
        @staticmethod
        def sample(uncertainty, sample_size):
            indexes = np.arange(len(uncertainty))
            uncertainty = uncertainty.astype('double')
            uncertainty[uncertainty<0] = 0
            probabilities = uncertainty / np.sum(uncertainty)
            return np.random.choice(indexes, sample_size, replace=False, p=probabilities)
