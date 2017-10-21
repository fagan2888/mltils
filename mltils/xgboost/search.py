
import numpy as np
import pandas as pd
import xgboost as xgb
from ..search.base import BoosterSearchBase


class XgbGridSearch(BoosterSearchBase):
    def __init__(self, param_grid, cv, metric, maximize):
        super().__init__(param_grid, cv, metric, maximize)

    def tr_transform(self, tr_x, tr_y):
        return xgb.DMatrix(tr_x, label=tr_y)

    def te_transform(self, te_x):
        return xgb.DMatrix(te_x)

    def eval_cv(self, xgb_cv_result, n_folds):
        metric_mean = '%s-mean' % self.metric
        metric_std = '%s-std' % self.metric
        cv_result = pd.DataFrame({
            metric_mean: xgb_cv_result['test-%s' % metric_mean],
            metric_std: xgb_cv_result['test-%s' % metric_std]
        })
        return self._eval_cv(cv_result, n_folds)

    def compute_booster_cv(self, params, tr_data, *args, **kwargs):
        cv_models = {}
        cv_result = xgb.cv(
            params,
            tr_data,
            maximize=self.maximize,
            folds=self.folds,
            callbacks=[save_cv_models(cv_models)],
            *args, **kwargs
        )
        return cv_result, cv_models['cv_packs']

    def extract_model_meta_features(self, comb_id, cv_packs, te_x):
        dtest = te_x
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
