
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def compute_auc_scores(X, target, verbose=True):
    auc_scores = []
    if verbose:
        iterator = tqdm(X, total=X.shape[1])
    else:
        iterator = X
    for var in iterator:
        auc = compute_auc(X[var], target)
        auc_scores.append({'var': var, 'auc': auc})
    return (
        pd.DataFrame(auc_scores)
          .sort_values(by='auc')
    )


def compute_random_auc_scores(X, target, n_iters=1000, verbose=True):
    if verbose:
        iterator = tqdm(range(n_iters))
    else:
        iterator = range(n_iters)
    return [
        sample_auc(X, target)
        for _ in iterator
    ]


def sample_auc(X, target):
    var = np.random.choice(X.columns)
    values = X[var].sample(X.shape[0]).values
    return compute_auc(values, target)


def compute_auc(x, target):
    auc = roc_auc_score(target, x)
    if auc < 0.5:
        auc = roc_auc_score(target, -x)
    return auc
