
import gc
import xgboost as xgb
from tqdm import tqdm


class BaggedXgboost(object):
    def __init__(self, n_models, verbose=True):
        self.n_models = n_models
        self.verbose = verbose
        self.models = []

    def train(self, params, dtrain, *args, **kwargs):
        if self.verbose:
            iterator = tqdm(range(self.n_models))
        else:
            iterator = range(self.n_models)
        for _ in iterator:
            self.models.append(xgb.train(params, dtrain, *args, **kwargs))
            gc.collect()
        return self

    def predict(self, dtest, *args, **kwargs):
        predictions = self.models[0].predict(dtest, *args, **kwargs)
        if self.verbose:
            iterator = tqdm(range(1, self.n_models))
        else:
            iterator = range(1, self.n_models)
        for i in iterator:
            predictions += self.models[i].predict(dtest, *args, **kwargs)
            gc.collect()
        predictions = predictions / self.n_models
        return predictions
