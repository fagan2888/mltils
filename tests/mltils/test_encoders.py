# pylint: disable=missing-docstring, invalid-name, import-error
import numpy as np
import pandas as pd

from mltils.encoders import DummyEncoder
from mltils.utils import nan_equal


def test_dummy_encoder_1():
    encoder = DummyEncoder()
    assert encoder is not None


def test_dummy_encoder_2():
    df = pd.DataFrame(
        {'A': ['a', 'b', 'c'],
         'B': ['d', 'd', 'f'],
         'C': [1, 2, 3]})
    encoded = DummyEncoder().fit_transform(df)
    assert encoded.shape == (3, 3 + 2 + 1)


def test_dummy_encoder_3():
    df = pd.DataFrame(
        {'A': ['a', 'b', np.nan],
         'B': ['d', 'd', 'f'],
         'C': [1, 2, np.nan]})
    encoded = DummyEncoder().fit_transform(df)
    assert encoded.shape == (3, 3 + 2 + 1)


def test_dummy_encoder_4():
    df = pd.DataFrame(
        {'A': ['a', 'b', np.nan],
         'B': ['d', 'd', 'f'],
         'C': [1, 2, np.nan]})
    denc = DummyEncoder(rpl_nan_cat='ab').fit(df)
    assert denc.var_names == ['A_a', 'A_ab', 'A_b', 'B_d', 'B_f', 'C']


def test_dummy_encoder_5():
    df = pd.DataFrame(
        {'A': ['a', 'b', np.nan],
         'B': ['d', 'd', 'f'],
         'C': [1, 2, np.nan]})
    encoded = DummyEncoder().fit_transform(df).todense().view(np.ndarray)
    expected = np.array(
        [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0],
         [1, 1, 0],
         [0, 0, 1],
         [1, 2, np.nan]]).T
    assert nan_equal(expected, encoded)
