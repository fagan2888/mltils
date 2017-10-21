
import numpy as np


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:,2], -1*all[:,1]))]
    total_losses = all[:,0].sum()
    gini_sum = all[:,0].cumsum().sum() / total_losses
    gini_sum -= (len(actual) + 1) / 2.
    return gini_sum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


def gini_normalized_xgb(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', gini_normalized(labels, preds)


def gini_normalized_lgb(preds, dtrain):
    labels = list(dtrain.get_label())
    return 'gini', gini_normalized(labels, preds), True
