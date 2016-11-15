# pylint: disable=missing-docstring, import-error
from __future__ import print_function

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from statsmodels.distributions import ECDF

from ..utils.generic_utils import validate_is_data_frame, _print, ReplacementManager
from .base import EncoderBase

# TODO:  - Better understand BaseEstimator
#        - Implement a decorator for parameter validation (pd.DataFrame expected)


class CountEncoder(EncoderBase):
    def __init__(self, nan_str_rpl='NaN', nan_num_rpl=-99999,
                 verbose=False):
        self.nenc = NanEncoder(
            str_rpl=nan_str_rpl, num_rpl=nan_num_rpl,
            ignore_numeric=False, copy=True, verbose=False)
        self.count_maps = {}
        self.verbose = verbose
        self.variables = None

    def fit(self, data, variables=None):
        validate_is_data_frame(data)
        self.variables = variables if variables is not None else data.columns
        data = data[self.variables]
        data = self.nenc.fit_transform(data)
        var_itr = self.get_var_itr(msg='Computing counts...')
        for var in var_itr:
            var_count = data[var].value_counts().to_dict()
            self.count_maps[var] = var_count
        return self

    def transform(self, data):
        validate_is_data_frame(data)
        data = data[self.variables]
        data = self.nenc.transform(data)
        count_vars = []
        nb_samples = data.shape[0]
        if self.verbose == 1:
            _print('Extracting counts...')
            cols_itr = tqdm(data.columns)
        else:
            cols_itr = data.columns
        for var in cols_itr:
            count_var_name = var + '_count'
            count_var = pd.Series(np.zeros(nb_samples), index=data.index,
                                  name=count_var_name)
            values = data[var]
            if self.verbose == 2:
                _print('Extracting counts for %s' % var)
                itr = tqdm(values.iteritems(), total=nb_samples)
            else:
                itr = values.iteritems()
            for index, current_value in itr:
                if current_value in self.count_maps[var]:
                    count_var.at[index] = self.count_maps[var][current_value]
            count_vars.append(count_var)
        return pd.concat(count_vars, axis=1)


class DummyEncoder(EncoderBase):
    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(self, sep='_', ifq_thrshld=0, ifq_num_rpl=-999,
                 ifq_str_rpl='__ifq__', nan_num_rpl=-99999, nan_str_rpl='NaN',
                 verbose=False, copy=True):
        self.ifq_thrshld = ifq_thrshld
        self.sep = sep
        self.copy = copy
        self.cat_vars = None
        self.num_vars = None
        self.var_names = []
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
        self.cat_enc = CategoryEncoder(
            ifq_thrshld=ifq_thrshld, ifq_num_rpl=ifq_num_rpl,
            ifq_str_rpl=ifq_str_rpl, nan_str_rpl=nan_str_rpl,
            nan_num_rpl=nan_num_rpl, copy=False)
        self.verbose = verbose
        self.variables = None

    def fit(self, data, variables=None):
        self.variables = data.columns if variables is None else variables
        self.cat_vars = _get_cat_vars_for(data)
        self.num_vars = np.setdiff1d(self.variables, self.cat_vars)
        if self.copy:
            data = data.copy()
        data = self.cat_enc.fit_transform(data)

        for var in self.cat_vars:
            unique_vals = self.cat_enc.var_values[var]
            self.var_names.extend(
                '%s%s%s' % (var, self.sep, str(value))
                for value in sorted(unique_vals))
        self.var_names.extend(self.num_vars)

        if self.verbose:
            _print('Fitting one hot encoder...')
        self.ohe.fit(data[self.cat_vars])
        if self.verbose:
            _print('Done!')
        return self

    def transform(self, data):
        # TODO: Do a better a check, showing the unexpected columns in the error
        if not data.columns.equals(self.variables):
            raise ValueError('Unexpected variables found!')
        if self.copy:
            data = data.copy()
        data = self.cat_enc.transform(data)
        ohe_enc = self.ohe.transform(data[self.cat_vars])
        num_data = data[self.num_vars]
        return sparse.hstack([ohe_enc, num_data], format='csr')


class CategoryEncoder(EncoderBase):
    # pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(self, ifq_thrshld=0, ifq_num_rpl=-999, ifq_str_rpl='__ifq__',
                 nan_num_rpl=-99999, nan_str_rpl='NaN', verbose=False,
                 copy=True):
        self.ifq_thrshld = ifq_thrshld
        self.unk_rpl_mgr = ReplacementManager(ifq_num_rpl, ifq_str_rpl)
        self.copy = copy
        self.lencs = {}
        self.var_values = {}
        self.nenc = NanEncoder(
            str_rpl=nan_str_rpl, num_rpl=nan_num_rpl,
            ignore_numeric=False, copy=False, verbose=False)
        self.ive = InfrequentValueEncoder(
            thrshld=ifq_thrshld, str_rpl=ifq_str_rpl,
            num_rpl=ifq_num_rpl, copy=False, verbose=False)
        self.verbose = verbose
        self.variables = None

    def fit(self, data, variables=None, all_vars=False):
        if variables is None:
            if all_vars:
                self.variables = data.columns
            else:
                self.variables = _get_cat_vars_for(data)
        else:
            self.variables = variables
        if self.copy:
            data = data.copy()
        data.loc[:, self.variables] = self.nenc.fit_transform(data, self.variables)
        data.loc[:, self.variables] = self.ive.fit_transform(data, self.variables)

        var_itr = self.get_var_itr(msg='Fitting category encoder...')
        for var in var_itr:
            unique_vals = set(data[var].unique())
            self.var_values[var] = unique_vals
            rpl_val = self.unk_rpl_mgr.get_rpl_for(data[var])
            values = np.concatenate([data[var].values, [rpl_val]])
            lenc = LabelEncoder().fit(values)
            self.lencs[var] = lenc
        return self

    def transform(self, data):
        if self.copy:
            data = data.copy()
        data.loc[:, self.variables] = self.nenc.transform(data[self.variables])
        data.loc[:, self.variables] = self.ive.transform(data[self.variables])

        var_itr = self.get_var_itr(msg='Encoding unknown values...')
        for var in var_itr:
            unique_vals = self.var_values[var]
            unknown_mask = ~(data[var].isin(unique_vals).values)
            if unknown_mask.any():
                rpl_val = self.unk_rpl_mgr.get_rpl_for(data[var])
                data.loc[unknown_mask, var] = rpl_val

        var_itr = self.get_var_itr(msg='Encoding categories...')
        for var in var_itr:
            lenc = self.lencs[var]
            data.loc[:, var] = lenc.transform(data[var])
        return data


class InfrequentValueEncoder(EncoderBase):
    # pylint: disable=too-many-arguments
    def __init__(self, thrshld=50, str_rpl='ifq__', num_rpl=-999,
                 copy=True, verbose=False):
        self.thrshld = thrshld
        self.rpl_mgr = ReplacementManager(num_rpl, str_rpl)
        self.copy = copy
        self.ifq_maps = {}
        self.known_maps = {}
        self.verbose = verbose
        self.variables = None

    def fit(self, data, variables=None):
        self.variables = data.columns if variables is None else variables
        if self.thrshld > 0:
            var_itr = self.get_var_itr(msg='Computing infrequent values...')
            for var in var_itr:
                var_count = data[var].value_counts()
                ifq_values = var_count.index[var_count <= self.thrshld]
                self.ifq_maps[var] = set(ifq_values)
                self.known_maps[var] = set(data[var].unique())
        return self

    def transform(self, data):
        if self.thrshld > 0:
            if self.copy:
                data = data.copy()
            var_itr = self.get_var_itr(msg='Encoding infrequent values...')
            for var in var_itr:
                ifq_values = self.ifq_maps[var]
                known_values = self.known_maps[var]
                ifq_rows = (data[var].isin(ifq_values)) | (~data[var].isin(known_values))
                ifq_rows = ifq_rows & (~data[var].isnull())
                if ifq_rows.any():
                    rpl_val = self.rpl_mgr.get_rpl_for(data[var])
                    data.loc[ifq_rows, var] = rpl_val
        return data


class PercentileEncoder(EncoderBase):
    def __init__(self, verbose=False):
        self.ecdfs = {}
        self.verbose = verbose
        self.variables = None

    def fit(self, data):
        self.variables = data.select_dtypes(include=[np.number]).columns
        var_itr = self.get_var_itr(msg='Fitting ECDFs...')
        for var in var_itr:
            self.ecdfs[var] = ECDF(data[var].values)
        return self

    def transform(self, data):
        var_itr = self.get_var_itr(msg='Extracting percentiles...')
        percentiles = []
        for var in var_itr:
            ecdf = self.ecdfs[var]
            prcntl_var_name = '%s_prctl' % var
            prcntl_var = pd.Series(ecdf(data[var].values), index=data.index,
                                   name=prcntl_var_name)
            percentiles.append(prcntl_var)
        extracted = pd.concat(percentiles, axis=1)
        return extracted


class NanEncoder(EncoderBase):
    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(self, str_rpl='NaN', num_rpl=-99999, copy=True,
                 ignore_numeric=True, verbose=False):
        self.rpl_mgr = ReplacementManager(num_rpl, str_rpl)
        self.copy = copy
        self.ignore_numeric = ignore_numeric
        self.verbose = verbose
        self.variables = None

    def fit(self, data, variables=None):
        if variables is None:
            variables = data.columns
        if self.ignore_numeric:
            variables = data[variables].select_dtypes(exclude=[np.number]).columns
        self.variables = variables
        return self

    def transform(self, data):
        if self.copy:
            data = data.copy()
        var_itr = self.get_var_itr(msg='Encoding NaN values...')
        for var in var_itr:
            null_mask = data[var].isnull()
            if null_mask.any():
                rpl_val = self.rpl_mgr.get_rpl_for(data[var])
                data.loc[null_mask, var] = rpl_val
        return data


def _get_cat_vars_for(data):
    return data.select_dtypes(include=['object']).columns
