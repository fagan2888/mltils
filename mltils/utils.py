# pylint: disable=missing-docstring, import-error
from __future__ import print_function
import sys
import numpy as np
import pandas as pd


def validate_is_data_frame(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame.')


def nan_equal(arr1, arr2):
    return ((arr1 == arr2) | (np.isnan(arr1) & np.isnan(arr2))).all()


def _print(msg):
    print(msg, file=sys.stderr)


class ReplacementManager(object):
    # pylint: disable=too-few-public-methods
    def __init__(self, num_rpl, str_rpl):
        self.num_rpl = num_rpl
        self.str_rpl = str_rpl

    def get_rpl_for(self, series):
        if series.dtype == 'object':
            return self.str_rpl
        else:
            return self.num_rpl
