
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ParameterGrid


class XgbGridSearch(object):
    def __init__(self, param_grid, cv, eval_cv, print_cv_info, print_params):
        self.param_grid = ParameterGrid(param_grid)
        self.cv = cv
        self.eval_cv = eval_cv
        self.print_cv_info = print_cv_info
        self.print_params = print_params
        self.folds = None
        self.n_train_rows = None
        self.gs_result = None

    def run(self, tr_x, tr_y, te_x, *args, **kwargs):
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dtest = xgb.DMatrix(te_x)
        self.folds = [fold for fold in self.cv.split(tr_x, tr_y)]
        self.n_train_rows = tr_x.shape[0]
        return self._run(dtrain, dtest, *args, **kwargs)

    def _run(self, dtrain, dtest, *args, **kwargs):
        self.gs_result = []
        n_combinations = len(self.param_grid)
        print()
        print('Number of params combinations: %d' % n_combinations)
        print()
        for i, params in enumerate(self.param_grid):
            comb_id = i + 1
            print('Starting cv for combination %d...\n' % comb_id)
            self.print_params(params)
            info = self._compute_xgb_cv(comb_id, params, dtrain, dtest,
                                        *args, **kwargs)
            self.print_cv_info(info)
            self.gs_result.append(info)
        return self.gs_result

    def _compute_xgb_cv(self, comb_id, params, dtrain, dtest, *args, **kwargs):
        cv_models = {}
        cv_res = xgb.cv(
            params,
            dtrain,
            folds=self.folds,
            callbacks=[save_cv_models(cv_models)],
            *args, **kwargs
        )
        n_folds = len(self.folds)
        info = self.eval_cv(cv_res, n_folds)
        info['comb_id'] = comb_id
        info['params'] = params
        info['cv_result'] = cv_res
        tr_meta_ftr, te_meta_ftr = self._extract_model_meta_ftrs(
            comb_id, cv_models['cv_packs'], dtest
        )
        info['tr_meta_ftr'] = tr_meta_ftr
        info['te_meta_ftr'] = te_meta_ftr
        gc.collect()
        return info

    def extract_meta_features(self):
        tr_meta_ftrs = []
        te_meta_ftrs = []
        for result in self.gs_result:
            tr_meta_ftrs.append(result['tr_meta_ftr'])
            te_meta_ftrs.append(result['te_meta_ftr'])
        tr_meta_ftrs = pd.concat(tr_meta_ftrs, axis=1)
        te_meta_ftrs = pd.concat(te_meta_ftrs, axis=1)
        return tr_meta_ftrs, te_meta_ftrs

    def _extract_model_meta_ftrs(self, comb_id, cv_packs, dtest):
        dtest.feature_names = None
        tr_meta_ftr = np.zeros(self.n_train_rows)
        te_meta_ftr = np.zeros(dtest.num_row())
        for cv_pack, fold in zip(cv_packs, self.folds):
            val_idx = fold[1]
            assert len(val_idx) == cv_pack.dtest.num_row()
            tr_meta_ftr[val_idx] = cv_pack.bst.predict(cv_pack.dtest)
            te_meta_ftr += cv_pack.bst.predict(dtest)
        te_meta_ftr = te_meta_ftr / len(self.folds)
        model_name = 'xgb_%d' % comb_id
        tr_meta_ftr = pd.Series(tr_meta_ftr, name=model_name)
        te_meta_ftr = pd.Series(te_meta_ftr, name=model_name)
        return tr_meta_ftr, te_meta_ftr


def save_cv_models(cv_models):
    def callback(env):
        cv_models['cv_packs'] = env.cvfolds
    return callback
