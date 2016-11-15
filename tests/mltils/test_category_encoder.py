# pylint: disable=missing-docstring, invalid-name, import-error
import numpy as np
import pandas as pd

from mltils.preprocessing.encoders import CategoryEncoder
from mltils.utils.test_utils import _test_immutability


def test_category_encoder_1():
    cenc = CategoryEncoder()
    assert cenc is not None


def test_category_encoder_2():
    df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', 'd']})
    encoded = CategoryEncoder().fit_transform(df)
    expected = pd.DataFrame({'A': [1, 1, 2, 3, 4]})
    assert expected.equals(encoded)


def test_category_encoder_3():
    df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', np.nan, np.nan, 'a']})
    cenc = CategoryEncoder(nan_str_rpl='d').fit(df)
    encoded = cenc.transform(df)
    expected = pd.DataFrame({'A': [1, 1, 2, 3, 4, 4, 1]})
    assert expected.equals(encoded)


def test_category_encoder_4():
    df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', np.nan],
                       'B': [1, 2, np.nan, np.nan, 3]})
    cenc = CategoryEncoder(nan_str_rpl='d').fit(df)
    encoded = cenc.transform(df)
    expected = pd.DataFrame({'A': [1, 1, 2, 3, 4],
                             'B': [1, 2, np.nan, np.nan, 3]})
    assert expected.equals(encoded)


def test_category_encoder_5():
    df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', np.nan],
                       'B': [1, 2, np.nan, np.nan, 3]})
    cenc = CategoryEncoder(nan_str_rpl='d', nan_num_rpl=-1)
    cenc.fit(df, all_vars=True)
    encoded = cenc.transform(df)
    expected = pd.DataFrame({'A': [1, 1, 2, 3, 4],
                             'B': [2, 3, 1, 1, 4]})
    assert expected.equals(encoded)


def test_category_encoder_6():
    tr_df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', np.nan]})
    cenc = CategoryEncoder(nan_str_rpl='_nan_', ifq_str_rpl='_ifq_')
    cenc.fit(tr_df)
    te_df = pd.DataFrame({'A': [np.nan, 'a', 'z', 'g', 'c']})
    encoded = cenc.transform(te_df)
    expected = pd.DataFrame({'A': [1, 2, 0, 0, 4]})
    assert expected.equals(encoded)


def test_category_encoder_7():
    tr_df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', np.nan]})
    cenc = CategoryEncoder(nan_str_rpl='_nan_', ifq_str_rpl='_ifq_',
                           ifq_thrshld=2)
    cenc.fit(tr_df)
    te_df = pd.DataFrame({'A': [np.nan, 'a', 'z', 'g', 'c']})
    encoded = cenc.transform(te_df)
    expected = pd.DataFrame({'A': [0, 0, 0, 0, 0]})
    assert expected.equals(encoded)


def test_category_encoder_8():
    tr_df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', np.nan, np.nan, np.nan]})
    cenc = CategoryEncoder(nan_str_rpl='_nan_', ifq_str_rpl='_ifq_',
                           ifq_thrshld=2)
    cenc.fit(tr_df)
    te_df = pd.DataFrame({'A': [np.nan, 'a', 'z', 'g', 'c']})
    encoded = cenc.transform(te_df)
    expected = pd.DataFrame({'A': [1, 0, 0, 0, 0]})
    assert expected.equals(encoded)


def test_category_encoder_9():
    tr_df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', np.nan],
                          'B': [1, 2, np.nan, np.nan, 1]})
    cenc = CategoryEncoder(nan_str_rpl='_nan_', ifq_str_rpl='_ifq_',
                           nan_num_rpl=-1, ifq_num_rpl=-2, ifq_thrshld=1)
    cenc.fit(tr_df, all_vars=False)
    te_df = pd.DataFrame({'A': ['a', np.nan, 'g', 'h', 'b'],
                          'B': [np.nan, np.nan, 1, 10, 11]})
    encoded = cenc.transform(te_df)
    expected = pd.DataFrame({'A': [1, 0, 0, 0, 0],
                             'B': [np.nan, np.nan, 1, 10, 11]})
    assert expected.equals(encoded)


def test_category_encoder_10():
    tr_df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', np.nan],
                          'B': [1, 2, np.nan, np.nan, 1]})
    cenc = CategoryEncoder(nan_str_rpl='_nan_', ifq_str_rpl='_ifq_',
                           nan_num_rpl=-1, ifq_num_rpl=-2, ifq_thrshld=1)
    cenc.fit(tr_df, all_vars=True)
    te_df = pd.DataFrame({'A': ['a', np.nan, 'g', 'h', 'b'],
                          'B': [np.nan, np.nan, 1, 10, 11]})
    encoded = cenc.transform(te_df)
    expected = pd.DataFrame({'A': [1, 0, 0, 0, 0],
                             'B': [1, 1, 2, 0, 0]})
    assert expected.equals(encoded)


def test_category_encoder_11():
    _test_immutability(encoder=CategoryEncoder())
