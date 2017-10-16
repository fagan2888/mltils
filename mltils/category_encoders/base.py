
from sklearn.base import BaseEstimator, TransformerMixin


class EncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cat_vars, encoder_class, suffix, *args, **kwargs):
        kwargs['cols'] = cat_vars
        self.cat_vars = cat_vars
        self.suffix = suffix
        self.encoder = encoder_class(*args, **kwargs)

    def fit(self, X, y=None, **kwargs):
        self.encoder.fit(X[self.cat_vars], y, **kwargs)

    def transform(self, X):
        encoded = self.encoder.transform(X[self.cat_vars])
        var_map = {var: var + '_' + self.suffix for var in encoded}
        encoded.rename(columns=var_map, inplace=True)
        return encoded
