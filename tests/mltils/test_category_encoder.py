# pylint: disable=missing-docstring, invalid-name, import-error
import numpy as np
import pandas as pd

from mltils.encoders import CategoryEncoder


def test_category_encoder_1():
    cenc = CategoryEncoder()
    assert cenc is not None


def test_category_encoder_2():
    df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', 'd']})
    encoded = CategoryEncoder().fit_transform(df)
    expected = pd.DataFrame({'A': [1, 1, 2, 3, 4]})
    assert expected.equals(encoded)


def test_category_encoder_3():
    df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', np.nan]})
    cenc = CategoryEncoder(nan_cat_rpl='d').fit(df)
    encoded = cenc.transform(df)
    expected = pd.DataFrame({'A': [1, 1, 2, 3, 4]})
    assert expected.equals(encoded)


def test_category_encoder_4():
    df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', np.nan],
                       'B': [1, 2, np.nan, np.nan, 3]})
    cenc = CategoryEncoder(nan_cat_rpl='d').fit(df)
    encoded = cenc.transform(df)
    expected = pd.DataFrame({'A': [1, 1, 2, 3, 4],
                             'B': [1, 2, np.nan, np.nan, 3]})
    assert expected.equals(encoded)


def test_category_encoder_5():
    df = pd.DataFrame({'A': ['a', 'a', 'b', 'c', np.nan],
                       'B': [1, 2, np.nan, np.nan, 3]})
    cenc = CategoryEncoder(nan_cat_rpl='d').fit(df)
    encoded = cenc.transform(df)
    expected = pd.DataFrame({'A': [1, 1, 2, 3, 4],
                             'B': [1, 2, np.nan, np.nan, 3]})
    assert expected.equals(encoded)
