import numpy as np
from math import log2


def uq_accuracy(uq, errors, percentile=0.1):
    """Shows intersection of worst by error/uq in percentile"""
    k = int(len(uq)*percentile)
    worst_uq = np.argsort(np.ravel(uq))[-k:]
    worst_error = np.argsort(np.ravel(errors))[-k:]
    return len(set(worst_uq).intersection(set(worst_error)))/k


def dcg(relevances, scores):
    """
    Discounting cumulative gain, metric of ranking quality
    For UQ - relevance is ~ error, scores is uq
    """
    relevances = np.ravel(relevances)
    scores = np.ravel(scores)

    ranking = np.argsort(scores)[::-1]
    metric = 0
    for rank, score_id in enumerate(ranking):
        metric += relevances[score_id] / log2(rank+2)
        
    return metric


def ndcg(errors, uq):
    """Normalized DCG. We norm fact DCG on ideal DCG score"""
    return dcg(errors, uq) / dcg(errors, errors)
