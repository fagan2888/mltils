# pylint: disable=missing-docstring, invalid-name, import-error
import numpy as np
import pandas as pd

from mltils.encoders import NanEncoder


def test_nan_encoder_1():
    nenc = NanEncoder()
    assert nenc is not None


def test_nan_encoder_2():
    df = pd.DataFrame({'A': [np.nan, 'a', 'b', 'c', np.nan]})
    encoded = NanEncoder(str_rpl='NaN').fit_transform(df)
    expected = pd.DataFrame({'A': ['NaN', 'a', 'b', 'c', 'NaN']})
    assert expected.equals(encoded)


def test_nan_encoder_3():
    df = pd.DataFrame({'A': [np.nan, 'a', 'b', 'c', np.nan],
                       'B': [np.nan, 'a', 'b', 'c', np.nan]})
    encoded = NanEncoder(str_rpl='NaN').fit_transform(df, variables=['A'])
    expected = pd.DataFrame({'A': ['NaN', 'a', 'b', 'c', 'NaN'],
                             'B': [np.nan, 'a', 'b', 'c', np.nan]})
    assert expected.equals(encoded)


def test_nan_encoder_4():
    df = pd.DataFrame({'A': [np.nan, 'a', 'b', 'c', np.nan],
                       'B': [1, 2, 2, np.nan, np.nan]})
    nenc = NanEncoder(str_rpl='NaN', ignore_numeric=True)
    encoded = nenc.fit_transform(df, variables=['A'])
    expected = pd.DataFrame({'A': ['NaN', 'a', 'b', 'c', 'NaN'],
                             'B': [1, 2, 2, np.nan, np.nan]})
    assert expected.equals(encoded)


def test_nan_encoder_5():
    df = pd.DataFrame({'A': [np.nan, 'a', 'b', 'c', np.nan],
                       'B': [1, 2, 2, np.nan, np.nan]})
    nenc = NanEncoder(str_rpl='NaN', num_rpl=-1, ignore_numeric=False)
    encoded = nenc.fit_transform(df)
    expected = pd.DataFrame({'A': ['NaN', 'a', 'b', 'c', 'NaN'],
                             'B': [1.0, 2.0, 2.0, -1.0, -1.0]})
    assert expected.equals(encoded)


def test_nan_encoder_6():
    df = pd.DataFrame({'A': [np.nan, 'a', 'b', 'c', np.nan],
                       'B': [1, 2, 2, np.nan, np.nan],
                       'C': [1, 2, 2, np.nan, np.nan]})
    nenc = NanEncoder(str_rpl='NaN', num_rpl=-1, ignore_numeric=True)
    encoded = nenc.fit_transform(df, variables=['A', 'B'])
    expected = pd.DataFrame({'A': ['NaN', 'a', 'b', 'c', 'NaN'],
                             'B': [1, 2, 2, np.nan, np.nan],
                             'C': [1, 2, 2, np.nan, np.nan]})
    assert expected.equals(encoded)
