# pylint: disable=missing-docstring, import-error
from __future__ import print_function

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from statsmodels.distributions import ECDF

from .utils import validate_is_data_frame, _print, ReplacementManager


class CountEncoder(object):
    def __init__(self, str_nan_rpl='NaN', num_nan_rpl=-99999, verbose=False):
        self.str_nan_rpl = str_nan_rpl
        self.num_nan_rpl = num_nan_rpl
        self.verbose = verbose
        self.nan_rpl_mgr = ReplacementManager(num_nan_rpl, str_nan_rpl)
        self.count_maps = {}
        self.variables = None

    def fit(self, data, variables=None):
        validate_is_data_frame(data)
        self.variables = variables if variables is not None else data.columns
        data = data[self.variables]
        if self.verbose > 0:
            _print('Computing counts...')
            var_itr = tqdm(data.columns)
        else:
            var_itr = data.columns
        for var in var_itr:
            rpl = self.nan_rpl_mgr.get_rpl_for(data[var])
            var_count = data[var].fillna(rpl).value_counts().to_dict()
            self.count_maps[var] = var_count
        return self

    def transform(self, data):
        validate_is_data_frame(data)
        data = data[self.variables]
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
            rpl = self.nan_rpl_mgr.get_rpl_for(data[var])
            values = data[var].fillna(rpl)
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

    def fit_transform(self, data, variables=None):
        self.fit(data, variables)
        return self.transform(data)


class DummyEncoder(object):
    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(self, infq_thrshld=0, sep='_', verbose=False, num_rpl=-99999,
                 str_rpl='__unknown__', nan_cat_rpl='NaN', copy=True):
        self.infq_thrshld = infq_thrshld
        self.sep = sep
        self.verbose = verbose
        self.copy = copy
        self.variables = None
        self.cat_vars = None
        self.num_vars = None
        self.var_names = []
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
        self.cat_enc = CategoryEncoder(
            infq_thrshld=infq_thrshld,
            num_rpl=num_rpl,
            str_rpl=str_rpl,
            nan_cat_rpl=nan_cat_rpl,
            copy=copy
        )

    def fit(self, data):
        validate_is_data_frame(data)
        self.variables = data.columns
        self.cat_vars = data.select_dtypes(include=['category', 'object']).columns
        self.num_vars = np.setdiff1d(self.variables, self.cat_vars)

        if self.copy:
            data = data.copy()

        data = self.cat_enc.fit_transform(data)

        for var in self.cat_vars:
            unique_vals = self.cat_enc.var_values[var]
            self.var_names.extend(
                '%s%s%s' % (var, self.sep, str(value))
                for value in sorted(unique_vals)
            )
        self.var_names.extend(self.num_vars)

        if self.verbose:
            _print('Fitting one hot encoder...')
        self.ohe.fit(data[self.cat_vars])
        if self.verbose:
            _print('Done!')
        return self

    def transform(self, data):
        validate_is_data_frame(data)
        if not data.columns.equals(self.variables):
            raise ValueError('Unexpected variables found!')

        if self.copy:
            data = data.copy()

        data = self.cat_enc.transform(data)
        ohe_enc = self.ohe.transform(data[self.cat_vars])
        num_data = data[self.num_vars]
        return sparse.hstack([ohe_enc, num_data], format='csr')

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class CategoryEncoder(object):
    # pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(self, infq_thrshld=0, num_rpl=-99999,
                 str_rpl='__unknown__', nan_cat_rpl='NaN', verbose=False,
                 copy=True):
        self.infq_thrshld = infq_thrshld
        self.rpl_mgr = ReplacementManager(num_rpl, str_rpl)
        self.nan_cat_rpl = nan_cat_rpl
        self.verbose = verbose
        self.copy = copy
        self.lencs = {}
        self.var_values = {}
        self.variables = None
        self.ive = InfrequentValueEncoder(
            thrshld=infq_thrshld, str_rpl=str_rpl,
            num_rpl=num_rpl, verbose=verbose)

    def fit(self, data, variables=None):
        if variables is None:
            self.variables = data.select_dtypes(include=['category', 'object']).columns
        if self.copy:
            data = data.copy()
        if self.infq_thrshld > 0:
            data.loc[:, self.variables] = self.ive.fit_transform(data, self.variables)
        if self.verbose:
            _print('Fitting category encoder...')
            var_itr = tqdm(self.variables)
        else:
            var_itr = self.variables
        for var in var_itr:
            unique_vals = set(data[var].unique())
            self.var_values[var] = unique_vals
            rpl_val = self.rpl_mgr.get_rpl_for(data[var])
            values = np.concatenate([data[var].values, [rpl_val]])
            lenc = LabelEncoder().fit(values)
            self.lencs[var] = lenc
        return self

    def transform(self, data):
        if self.copy:
            data = data.copy()
        if self.infq_thrshld > 0:
            data.loc[:, self.variables] = self.ive.transform(data[self.variables])
        if self.verbose:
            _print('Encoding unknown values...')
            var_itr = tqdm(self.variables)
        else:
            var_itr = self.variables
        for var in var_itr:
            data = self.fill_na(data, var)
            unique_vals = self.var_values[var]
            unknown_mask = ~(data[var].isin(unique_vals).values)
            if unknown_mask.any():
                rpl_val = self.rpl_mgr.get_rpl_for(data[var])
                data.loc[unknown_mask, var] = rpl_val
        if self.verbose:
            _print('Extracting categories...')
        for var in var_itr:
            lenc = self.lencs[var]
            data.loc[:, var] = lenc.transform(data[var])
        return self

    def fit_transform(self, data, variables=None):
        self.fit(data, variables)
        return self.transform(data)

    def fill_na(self, data, var):
        null_mask = data[var].isnull()
        if null_mask.any():
            data.loc[null_mask, var] = self.nan_cat_rpl
        return data


class InfrequentValueEncoder(object):
    # pylint: disable=too-many-arguments
    def __init__(self, thrshld=50, str_rpl='__infrequent__',
                 num_rpl=-99999, verbose=False):
        self.thrshld = thrshld
        self.verbose = verbose
        self.nan_rpl_mgr = ReplacementManager(num_rpl, str_rpl)
        self.variables = None
        self.ifq_maps = {}

    def fit(self, data, variables=None):
        self.variables = data.columns if variables is None else variables
        if self.thrshld > 0:
            if self.verbose:
                _print('Computing infrequent values...')
                var_itr = tqdm(self.variables)
            else:
                var_itr = self.variables
            for var in var_itr:
                var_count = data[var].value_counts()
                ifq_values = var_count.index[var_count <= self.thrshld]
                self.ifq_maps[var] = set(ifq_values)
        return self

    def transform(self, data):
        if self.thrshld > 0:
            data = data[self.variables].copy()
            if self.verbose:
                _print('Encoding infrequent values...')
                var_itr = tqdm(self.variables)
            else:
                var_itr = self.variables
            for var in var_itr:
                ifq_values = self.ifq_maps[var]
                ifq_rows = data[var].isin(ifq_values)
                if ifq_rows.any():
                    rpl_val = self.nan_rpl_mgr.get_rpl_for(data[var])
                    data.loc[ifq_rows, var] = rpl_val
        return data

    def fit_transform(self, data, variables=None):
        self.fit(data, variables)
        return self.transform(data)


class PercentileEncoder(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.variables = None
        self.ecdfs = {}

    def fit(self, data):
        self.variables = data.select_dtypes(include=['float', 'int']).columns
        if self.verbose:
            _print('Fitting ECDFs...')
            var_itr = tqdm(self.variables)
        else:
            var_itr = self.variables
        for var in var_itr:
            self.ecdfs[var] = ECDF(data[var].values)
        return self

    def transform(self, data):
        if self.verbose:
            _print('Extracting percentiles...')
            var_itr = tqdm(self.variables)
        else:
            var_itr = self.variables
        percentiles = []
        for var in var_itr:
            ecdf = self.ecdfs[var]
            prcntl_var_name = '%s_prctl' % var
            prcntl_var = pd.Series(ecdf(data[var].values), index=data.index,
                                   name=prcntl_var_name)
            percentiles.append(prcntl_var)
        extracted = pd.concat(percentiles, axis=1)
        return extracted

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
