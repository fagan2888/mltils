# # pylint: disable=missing-docstring, invalid-name, import-error
# import numpy as np
# import pandas as pd
#
# from mltils.encoders import DummyEncoder
# from mltils.utils import nan_equal
#
#
# def test_dummy_encoder_1():
#     denc = DummyEncoder()
#     assert denc is not None
#
#
# def test_dummy_encoder_2():
#     df = pd.DataFrame(
#         {'A': ['a', 'b', 'c'],
#          'B': ['d', 'd', 'f'],
#          'C': [1, 2, 3]})
#     encoded = DummyEncoder().fit_transform(df)
#     assert encoded.shape == (3, 3 + 2 + 1)
#
#
# def test_dummy_encoder_3():
#     df = pd.DataFrame(
#         {'A': ['a', 'b', np.nan],
#          'B': ['d', 'd', 'f'],
#          'C': [1, 2, np.nan]})
#     encoded = DummyEncoder().fit_transform(df)
#     assert encoded.shape == (3, 3 + 2 + 1)
#
#
# def test_dummy_encoder_4():
#     df = pd.DataFrame(
#         {'A': ['a', 'b', np.nan],
#          'B': ['d', 'd', 'f'],
#          'C': [1, 2, np.nan]})
#     denc = DummyEncoder(nan_cat_rpl='ab').fit(df)
#     assert denc.var_names == ['A_a', 'A_ab', 'A_b', 'B_d', 'B_f', 'C']
#
#
# def test_dummy_encoder_5():
#     df = pd.DataFrame(
#         {'A': ['a', 'b', np.nan],
#          'B': ['d', 'd', 'f'],
#          'C': [1, 2, np.nan]})
#     encoded = DummyEncoder().fit_transform(df).todense().view(np.ndarray)
#     expected = np.array(
#         [[0, 0, 1],
#          [1, 0, 0],
#          [0, 1, 0],
#          [1, 1, 0],
#          [0, 0, 1],
#          [1, 2, np.nan]]).T
#     assert nan_equal(expected, encoded)
#
#
# def test_dummy_encoder_6():
#     df = pd.DataFrame(
#         {'A': ['a', 'b', 'd'],
#          'B': ['d', 'd', 'f'],
#          'C': [1, 2, np.nan]})
#     denc = DummyEncoder(infq_thrshld=1, str_rpl='unk').fit(df)
#     assert denc.var_names == ['A_unk', 'B_d', 'B_unk', 'C']
#
#
# def test_dummy_encoder_7():
#     tr_df = pd.DataFrame({'A': ['a', 'b', 'd']})
#     denc = DummyEncoder(infq_thrshld=0).fit(tr_df)
#     te_df = pd.DataFrame({'A': ['f', 'h', 'y']})
#     encoded = denc.transform(te_df).todense()
#     expected = np.array(
#         [[0, 0, 0],
#          [0, 0, 0],
#          [0, 0, 0]]).T
#     assert np.array_equal(expected, encoded)
#
#
# def test_dummy_encoder_8():
#     tr_df = pd.DataFrame({'A': ['a', 'b', 'd']})
#     denc = DummyEncoder(infq_thrshld=1).fit(tr_df)
#     te_df = pd.DataFrame({'A': ['f', 'h', 'y']})
#     encoded = denc.transform(te_df).todense()
#     expected = np.array([[1, 1, 1]]).T
#     assert np.array_equal(expected, encoded)
#
#
# def test_dummy_encoder_9():
#     df = pd.DataFrame(
#         {'A': ['a', np.nan, np.nan],
#          'B': ['c', 'c', 'd'],
#          'C': [1, 2, np.nan]})
#     df_before = df.copy()
#     _ = DummyEncoder().fit(df)
#     assert df_before.equals(df)
