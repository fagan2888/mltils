# pylint: disable=missing-docstring, invalid-name, import-error
import pandas as pd

from mltils.preprocessing.encoders import PercentileEncoder
from mltils.utils.test_utils import _test_immutability


def test_percentile_encoder_1():
    penc = PercentileEncoder()
    assert penc is not None


def test_percentile_encoder_2():
    df = pd.DataFrame({'A': [1., 2., 2., 3., 3.],
                       'B': [1, 2, 3, 4, 5]})
    encoded = PercentileEncoder().fit_transform(df)
    assert encoded.columns.tolist() == ['A_prctl', 'B_prctl']


def test_percentile_encoder_3():
    df = pd.DataFrame({'A': [1., 2.],
                       'B': [1, 2],
                       'C': ['A', 'B'],
                       'D': ['C', 'D']})
    encoded = PercentileEncoder().fit_transform(df)
    assert encoded.columns.tolist() == ['A_prctl', 'B_prctl']


def test_percentile_encoder_4():
    _test_immutability(encoder=PercentileEncoder())
