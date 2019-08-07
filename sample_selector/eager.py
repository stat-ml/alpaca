import numpy as np


class EagerSampleSelector:
    """
    Move samples from pool dataset to train dataset
    Takes samples with highest uncertainty first
    """
    def update_sets(self, X_train, y_train, X_pool, ue_values, sample_size, oracle):
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
    def sample(values, sample_size):
        return np.argsort(values)[::-1][:sample_size]
