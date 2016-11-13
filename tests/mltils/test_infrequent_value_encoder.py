# pylint: disable=missing-docstring, invalid-name, import-error
import pandas as pd

from mltils.encoders import InfrequentValueEncoder


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
