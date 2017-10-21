

import random
import numpy as np
from sklearn.base import BaseEstimator, clone
from tqdm import tqdm


class BaggedModel(BaseEstimator):
    def __init__(self, estimator, n_models, random_state, verbose=True):
        self.estimator = estimator
        self.n_models = n_models
        self.random_state = random_state
        self.verbose = verbose
        self.models = []

    def fit(self, X, y, *args, **kwargs):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        if self.verbose:
            iterator = tqdm(range(self.n_models))
        else:
            iterator = range(self.n_models)
        for _ in iterator:
            model = clone(self.estimator)
            self.models.append(model.fit(X, y, *args, **kwargs))
        return self

    def predict_proba(self, X, *args, **kwargs):
        predictions = self.models[0].predict_proba(X, *args, **kwargs)
        for i in range(1, self.n_models):
            predictions += self.models[i].predict_proba(X, *args, **kwargs)
        predictions = predictions / self.n_models
        return predictions
