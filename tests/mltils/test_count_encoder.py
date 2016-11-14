# pylint: disable=missing-docstring, invalid-name, import-error
import numpy as np
import pandas as pd

from mltils.preprocessing.encoders import CountEncoder


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


def _test_immutability(encoder):
    df = pd.DataFrame(
        {'A': ['a', np.nan, np.nan],
         'B': ['c', 'c', 'd'],
         'C': [1, 2, np.nan]})
    df_before = df.copy()
    _ = encoder.fit_transform(df)
    assert df_before.equals(df)


def test_count_encoder_4():
    _test_immutability(encoder=CountEncoder())


def test_count_encoder_5():
    df = pd.DataFrame(
        {'A': ['a', np.nan, np.nan],
         'B': ['c', 'c', 'd'],
         'C': [1, 2, np.nan]})
    encoded = CountEncoder().fit_transform(df, variables=['A', 'B'])
    expected = pd.DataFrame(
        {'A_count': [1., 2., 2.],
         'B_count': [2., 2., 1.]})
    assert expected.equals(encoded)
