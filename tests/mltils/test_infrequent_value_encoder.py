# pylint: disable=missing-docstring, invalid-name, import-error
import numpy as np
import pandas as pd

from mltils.preprocessing.encoders import InfrequentValueEncoder


def test_infrequent_value_encoder_1():
    ive = InfrequentValueEncoder()
    assert ive is not None


def test_infrequent_value_encoder_2():
    df = pd.DataFrame({'A': ['a', 'a', 'b', 'b', 'c']})
    ive = InfrequentValueEncoder(thrshld=1, str_rpl='ifq')
    encoded = ive.fit_transform(df)
    expected = pd.DataFrame({'A': ['a', 'a', 'b', 'b', 'ifq']})
    assert expected.equals(encoded)


def test_infrequent_value_encoder_3():
    df = pd.DataFrame({'A': ['a', 'a', 'b', 'b', 'c']})
    ive = InfrequentValueEncoder(thrshld=0, str_rpl='ifq')
    encoded = ive.fit_transform(df)
    expected = pd.DataFrame({'A': ['a', 'a', 'b', 'b', 'c']})
    assert expected.equals(encoded)


def test_infrequent_value_encoder_4():
    df = pd.DataFrame({'A': ['a', 'a', 'b', 'b', 'c'],
                       'B': [1, 1, 1, 2, 3]})
    ive = InfrequentValueEncoder(thrshld=2, str_rpl='ifq', num_rpl=-1)
    encoded = ive.fit_transform(df)
    expected = pd.DataFrame({'A': ['ifq', 'ifq', 'ifq', 'ifq', 'ifq'],
                             'B': [1, 1, 1, -1, -1]})
    assert expected.equals(encoded)


def test_infrequent_value_encoder_5():
    tr_df = pd.DataFrame({'A': ['a', 'a', 'b', 'b', 'c']})
    ive = InfrequentValueEncoder(thrshld=1, str_rpl='ifq')
    ive.fit(tr_df)
    te_df = pd.DataFrame({'A': ['c', 'd', 'e', 'a', 'b']})
    encoded = ive.transform(te_df)
    expected = pd.DataFrame({'A': ['ifq', 'ifq', 'ifq', 'a', 'b']})
    assert expected.equals(encoded)


def test_infrequent_value_encoder_6():
    tr_df = pd.DataFrame({'A': ['a', 'a', 'b', 'b', 'c', np.nan]})
    ive = InfrequentValueEncoder(thrshld=1, str_rpl='ifq')
    ive.fit(tr_df)
    te_df = pd.DataFrame({'A': [np.nan, 'c', 'd', 'e', 'a', 'b']})
    encoded = ive.transform(te_df)
    expected = pd.DataFrame({'A': [np.nan, 'ifq', 'ifq', 'ifq', 'a', 'b']})
    assert expected.equals(encoded)


def test_infrequent_value_encoder_7():
    df = pd.DataFrame({'A': [1, 2, 3, np.nan, 4, np.nan]})
    encoded = InfrequentValueEncoder(thrshld=1, num_rpl=-1).fit_transform(df)
    expected = pd.DataFrame({'A': [-1, -1, -1, np.nan, -1, np.nan]})
    assert expected.equals(encoded)
