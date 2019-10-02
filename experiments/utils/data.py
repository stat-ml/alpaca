from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import KFold


def scale(train, val):
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    val = scaler.transform(val)
    return train, val, scaler


def split_ood(x_all, y_all, percentile=10):
    threshold = np.percentile(y_all, percentile)
    ood_idx = np.argwhere(y_all > threshold)[:, 0]
    x_ood, y_ood = x_all[ood_idx], y_all[ood_idx]
    train_idx = np.argwhere(y_all <= threshold)[:, 0]
    x_train, y_train = x_all[train_idx], y_all[train_idx]

    return x_train, y_train, x_ood, y_ood


def multiple_kfold(k, data_size, max_iterations):
    kfold = KFold(k)
    for i in range(max_iterations):
        if i % k == 0:
            data_idx = np.random.permutation(data_size)
            idx_generator = kfold.split(data_idx)
        train_idx, val_idx = next(idx_generator)
        yield data_idx[train_idx], data_idx[val_idx]
