
import numpy as np
import pandas as pd
import lightgbm as lgb
from ..search.base import BoosterSearchBase


class LgbGridSearch(BoosterSearchBase):
    def __init__(self, param_grid, cv, metric, maximize):
        super().__init__(param_grid, cv, metric, maximize)

    def tr_transform(self, tr_x, tr_y):
        return lgb.Dataset(tr_x, tr_y)

    def te_transform(self, te_x):
        return te_x

    def eval_cv(self, cv_result, n_folds):
        cv_result = pd.DataFrame(cv_result.copy())
        cv_result.columns = [col.replace('stdv', 'std') for col in cv_result]
        return self._eval_cv(cv_result, n_folds)

    def compute_booster_cv(self, params, tr_data, *args, **kwargs):
        cv_models = {}
        cv_result = lgb.cv(
            params,
            tr_data,
            folds=self.folds,
            callbacks=[save_cv_models(cv_models)],
            *args, **kwargs
        )
        return cv_result, cv_models['cv_packs']

    def extract_model_meta_features(self, comb_id, cv_packs, te_x):
        tr_meta_ftr = np.zeros(self.n_train_rows)
        te_meta_ftr = np.zeros(te_x.shape[0])
        for lgb_model, fold in zip(cv_packs.boosters, self.folds):
            val_idx = fold[1]
            assert len(lgb_model.valid_sets) == 1
            assert len(val_idx) == lgb_model.valid_sets[0].num_data()
            tr_meta_ftr[val_idx] = lgb_model._Booster__inner_predict(data_idx=1)
            te_meta_ftr += lgb_model.predict(te_x)
        te_meta_ftr = te_meta_ftr / len(self.folds)
        model_name = 'lgb_%d' % comb_id
        tr_meta_ftr = pd.Series(tr_meta_ftr, name=model_name)
        te_meta_ftr = pd.Series(te_meta_ftr, name=model_name)
        return tr_meta_ftr, te_meta_ftr


def save_cv_models(cv_models):
    def callback(env):
        cv_models['cv_packs'] = env.model
    return callback
