# pylint: disable=missing-docstring, invalid-name, import-error
import numpy as np
import pandas as pd


def _test_immutability(encoder):
    df = pd.DataFrame(
        {'A': ['a', np.nan, np.nan],
         'B': ['c', 'c', 'd'],
         'C': [1, 2, np.nan]})
    df_before = df.copy()
    _ = encoder.fit_transform(df)
    assert df_before.equals(df)
