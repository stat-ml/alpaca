import numpy as np
from math import log2
from scipy.stats import percentileofscore


def get_uq_metrics(estimations, errors, acc_percentile=0.1, **kwargs):
    acc = uq_accuracy(estimations, errors, acc_percentile)
    ndcg = uq_ndcg(errors, estimations, **kwargs)
    ll = uq_ll(errors, estimations)
    return acc, ndcg, ll


def uq_accuracy(uq, errors, percentile=0.1):
    """Shows intersection of worst by error/uq in percentile"""
    k = int(len(uq)*percentile)
    worst_uq = np.argsort(np.ravel(uq))[-k:]
    worst_error = np.argsort(np.ravel(errors))[-k:]
    return len(set(worst_uq).intersection(set(worst_error)))/k


def dcg(relevances, scores, k):
    """
    Discounting cumulative gain, metric of ranking quality
    For UQ - relevance is ~ error, scores is uq
    """
    relevances = np.ravel(relevances)
    scores = np.ravel(scores)

    ranking = np.argsort(scores)[::-1]
    metric = 0
    for rank, score_id in enumerate(ranking[:k]):
        metric += relevances[score_id] / log2(rank+2)

    return metric


def ndcg(relevances, scores):
    """
    Normalized DCG. We norm fact DCG on ideal DCG score
    expect relevances, scores to be numpy ndarrays
    """
    k = sum(relevances != 0)
    return dcg(relevances, scores, k) / dcg(relevances, relevances, k)


def uq_ndcg(errors, uq, bins=None):
    """
    In UQ we care most of top erros,
    so we restructure errors to give top errors bigger relevance
    """
    if bins is None:
        bins = [80, 95, 99]

    sorted_errors = sorted(errors)
    errors_percentiles = [percentileofscore(sorted_errors, error) for error in errors]
    errors_digitized = np.digitize(errors_percentiles, bins)

    return ndcg(errors_digitized, uq)


def uq_ll(errors, uq):
    errors = np.ravel(errors)
    uq_squared = np.square(np.ravel(uq)) + 1e-10
    return -np.mean(np.log(uq_squared)/2 + np.square(errors)/2/uq_squared)
