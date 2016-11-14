# pylint: disable=missing-docstring, invalid-name, import-error
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.generic_utils import _print


class EncoderBase(BaseEstimator, TransformerMixin):
    # pylint: disable=too-few-public-methods
    def get_var_itr(self, msg):
        if self.verbose:
            _print(msg)
            var_itr = tqdm(self.variables)
        else:
            var_itr = self.variables
        return var_itr
