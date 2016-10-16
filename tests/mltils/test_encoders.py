# pylint: disable=missing-docstring, invalid-name, import-error
import numpy as np
import pandas as pd

from mltils.encoders import CountEncoder, DummyEncoder
from mltils.utils import nan_equal


def test_count_encoder_1():
    cenc = CountEncoder()
    assert cenc is not None


def test_count_encoder_2():
    df = pd.DataFrame(
        {'A': ['a', 'a', 'a'],
         'B': ['c', 'c', 'd'],
         'C': [1, 2, 3]})
    encoded = CountEncoder().fit_transform(df)
    expected = pd.DataFrame(
        {'A_count': [3., 3., 3.],
         'B_count': [2., 2., 1.],
         'C_count': [1., 1., 1.]})
    assert expected.equals(encoded)


def test_count_encoder_3():
    df = pd.DataFrame(
        {'A': ['a', np.nan, np.nan],
         'B': ['c', 'c', 'd'],
         'C': [1, 2, np.nan]})
    encoded = CountEncoder().fit_transform(df)
    expected = pd.DataFrame(
        {'A_count': [1., 2., 2.],
         'B_count': [2., 2., 1.],
         'C_count': [1., 1., 1.]})
    assert expected.equals(encoded)


def test_count_encoder_4():
    df = pd.DataFrame(
        {'A': ['a', np.nan, np.nan],
         'B': ['c', 'c', 'd'],
         'C': [1, 2, np.nan]})
    df_before = df.copy()
    _ = CountEncoder().fit_transform(df)
    assert df_before.equals(df)


def test_count_encoder_5():
    df = pd.DataFrame(
        {'A': ['a', np.nan, np.nan],
         'B': ['c', 'c', 'd'],
         'C': [1, 2, np.nan]})
    encoded = CountEncoder(variables=['A', 'B']).fit_transform(df)
    expected = pd.DataFrame(
        {'A_count': [1., 2., 2.],
         'B_count': [2., 2., 1.]})
    assert expected.equals(encoded)



def test_dummy_encoder_1():
    denc = DummyEncoder()
    assert denc is not None


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
    denc = DummyEncoder(nan_cat_rpl='ab').fit(df)
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
