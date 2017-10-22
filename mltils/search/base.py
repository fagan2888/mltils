
import os
import gc
import numpy as np
import pandas as pd
from abc import abstractmethod
from sklearn.model_selection import ParameterGrid


class BoosterSearchBase(object):
    def __init__(self, param_grid, cv, metric, maximize, save_partial=True):
        self.param_grid = ParameterGrid(param_grid)
        self.cv = cv
        self.metric = metric
        self.maximize = maximize
        self.save_partial = save_partial
        self.folds = None
        self.n_train_rows = None
        self.gs_result = None

    def run(self, tr_x, tr_y, te_x, *args, **kwargs):
        self.folds = [fold for fold in self.cv.split(tr_x, tr_y)]
        self.n_train_rows = tr_x.shape[0]
        tr_data = self.tr_transform(tr_x, tr_y)
        te_data = self.te_transform(te_x)
        self._run(tr_data, te_data, *args, **kwargs)

    @abstractmethod
    def tr_transform(self, tr_x):
        pass

    @abstractmethod
    def te_transform(self, te_x):
        pass

    def _run(self, tr_data, te_data, *args, **kwargs):
        self.gs_result = []
        n_combinations = len(self.param_grid)
        print('\nNumber of params combinations: %d\n' % n_combinations)
        for i, params in enumerate(self.param_grid):
            comb_id = i + 1
            print('Starting cv for combination %d...\n' % comb_id)
            print_params(params)
            cv_result, cv_packs = self.compute_booster_cv(
                params, tr_data, *args, **kwargs
            )
            n_folds = len(self.folds)
            info = self.eval_cv(cv_result, n_folds)
            info['comb_id'] = comb_id
            info['params'] = params
            info['cv_result'] = cv_result
            tr_meta_feature, te_meta_feature = self.extract_model_meta_features(
                comb_id, cv_packs, te_data
            )
            if self.save_partial:
                _save_partial(tr_meta_feature, te_meta_feature)

            info['tr_meta_feature'] = tr_meta_feature
            info['te_meta_feature'] = te_meta_feature
            gc.collect()
            print_cv_info(info, self.metric, self.maximize)
            self.gs_result.append(info)
        return self.gs_result

    @abstractmethod
    def eval_cv(self, cv_result, n_folds):
        pass

    def _eval_cv(self, cv_result, n_folds):
        metric_mean = '%s-mean' % self.metric
        metric_std = '%s-std' % self.metric
        metric_score = '%s-score' % self.metric
        if self.maximize:
            scores = cv_result[metric_mean] - cv_result[metric_std]
            best_idx = scores.argmax()
        else:
            scores = cv_result[metric_mean] + cv_result[metric_std]
            best_idx = scores.argmin()
        n_rounds = best_idx + 1
        n_rounds_aug = int(n_rounds * (n_folds / (n_folds - 1)))
        best_score = scores[best_idx]
        metric_mean_value = cv_result.at[best_idx, metric_mean]
        metric_std_value = cv_result.at[best_idx, metric_std]
        return {
            'n_rounds': n_rounds,
            'n_rounds_aug': n_rounds_aug,
            metric_score: best_score,
            metric_mean: metric_mean_value,
            metric_std: metric_std_value
        }

    @abstractmethod
    def compute_booster_cv(self, params, tr_data, *args, **kwargs):
        pass

    @abstractmethod
    def extract_model_meta_features(self, comb_id, cv_packs, te_data):
        pass

    def extract_meta_features(self):
        tr_meta_features = []
        te_meta_features = []
        for result in self.gs_result:
            tr_meta_features.append(result['tr_meta_feature'])
            te_meta_features.append(result['te_meta_feature'])
        tr_meta_features = pd.concat(tr_meta_features, axis=1)
        te_meta_features = pd.concat(te_meta_features, axis=1)
        return tr_meta_features, te_meta_features


def print_params(params):
    print(pd.Series(params).to_string())
    print()


def print_cv_info(cv_info, metric, maximize):
    metric_mean = '%s-mean' % metric
    metric_std = '%s-std' % metric
    metric_score = '%s-score' % metric
    print()
    print('----------------------------------------')
    print('Result combination %d\n' % cv_info['comb_id'])
    params = cv_info['params']
    print_params(params)
    print('%s: %f +- %f' % (
        metric, cv_info[metric_mean], cv_info[metric_std])
    )
    if maximize:
        print('score (mean - std): %f\n' % cv_info[metric_score])
    else:
        print('score (mean + std): %f\n' % cv_info[metric_score])
    print('----------------------------------------')
    print()


def _save_partial(tr_meta_feature, te_meta_feature):
    if not os.path.exists('meta_features'):
        os.makedirs('meta_features')
    if not os.path.exists('meta_features/partial'):
        os.makedirs('meta_features/partial')

    tr_path = 'meta_features/partial/tr_%s.csv' % tr_meta_feature.name
    te_path = 'meta_features/partial/te_%s.csv' % te_meta_feature.name
    tr_meta_feature.to_frame().to_csv(tr_path, index=False)
    te_meta_feature.to_frame().to_csv(te_path, index=False)
