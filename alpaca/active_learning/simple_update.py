import numpy as np
import torch
import torch.nn.functional as F
from dppy.finite_dpps import FiniteDPP


from alpaca.uncertainty_estimator import build_estimator
from alpaca.uncertainty_estimator.masks import build_mask


def update_set(x_pool, x_train, y_pool, y_train, step, method='mc_dropout', model=None, nn_runs=100, task='classification'):
    if task == 'classification':
        samples = torch.FloatTensor(x_pool).cuda()
    else:
        samples = torch.DoubleTensor(x_pool).cuda()

    if method == 'random':
        idxs = range(step)
    elif method == 'AL_dpp':
        mask = build_mask('mc_dropout')
        estimator = build_estimator(
            'bald_masked', model, dropout_mask=mask, num_classes=10,
            keep_runs=True, nn_runs=nn_runs)
        estimator.estimate(samples)  # to generate mcd
        mcd = estimator.last_mcd_runs().reshape(-1, nn_runs * 10)
        dpp = FiniteDPP('likelihood', **{'L': np.corrcoef(mcd)})
        idxs = set()
        while len(idxs) < step:
            dpp.sample_exact()
            idxs.update(dpp.list_of_samples[-1])
        idxs = list(idxs)[:step]
    elif method == 'error_oracle':
        predictions = F.softmax(model(samples), dim=1).detach().cpu().numpy()
        errors = -np.log(predictions[np.arange(len(predictions)),  y_pool])
        idxs = np.argsort(errors)[::-1][:step]
    elif method == 'stoch_oracle':
        predictions = F.softmax(model(samples), dim=1).detach().cpu().numpy()
        errors = -np.log(predictions[np.arange(len(predictions)), y_pool])
        idxs = np.random.choice(len(predictions), step, replace=False, p=errors/sum(errors))
    elif method == 'max_entropy':
        predictions = F.softmax(model(samples), dim=1).detach().cpu().numpy()
        entropies = entropy(predictions)
        idxs = np.argsort(entropies)[::-1][:step]
    else:
        if task == 'classification':
            estimator = build_estimator('bald_masked', model, dropout_mask=method, num_classes=10, nn_runs=nn_runs)
        else:
            estimator = build_estimator('mcdue_masked', model, dropout_mask=method, nn_runs=nn_runs)
        estimations = estimator.estimate(samples)
        idxs = np.argsort(estimations)[::-1][:step]
        estimator.reset()

    x_add, y_add = np.copy(x_pool[idxs]), np.copy(y_pool[idxs])
    x_train = np.concatenate((x_train, x_add))
    y_train = np.concatenate((y_train, y_add))
    x_pool = np.delete(x_pool, idxs, axis=0)
    y_pool = np.delete(y_pool, idxs, axis=0)

    return x_pool, x_train, y_pool, y_train


def entropy(x):
    return -np.sum(x * np.log(np.clip(x, 1e-8, 1)), axis=-1)
