#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import inspect
import sparse
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from xgboost import  XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

class TreeRegression(object):
    def __init__(self, mode, n_fold=10, seed=2019, save=False):
        self.mode = mode
        self.n_fold = n_fold
        self.seed = seed
        self.save = save
        self._check_mode(self.mode)

    @staticmethod
    def _check_mode(mode):
        assert mode in ['lgb', 'xgb', 'rf', 'ctb', 'ada', 'gbdt']

    def _get_gbm(self, params):
        if self.mode == 'lgb':
            gbm = LGBMRegressor(**params)
        elif self.mode == 'xgb':
            gbm = XGBRegressor(**params)
        else:
            raise ValueError()
        return gbm

    @staticmethod
    def _get_iteration_kwargs(gbm):
        predict_args = inspect.getfullargspec(gbm.predict).args
        if hasattr(gbm, 'best_iteration_'):
            best_iteration = getattr(gbm, 'best_iteration_')
            if 'num_iteration' in predict_args:
                iteration_kwargs = {'num_iteration': best_iteration}
            elif 'ntree_end' in predict_args:
                iteration_kwargs = {'ntree_end': best_iteration}
            else:
                raise ValueError()
        elif hasattr(gbm, 'best_ntree_limit'):
            best_iteration = getattr(gbm, 'best_ntree_limit')
            if 'ntree_limit' in predict_args:
                iteration_kwargs = {'ntree_limit': best_iteration}
            else:
                raise ValueError()
        else:
            raise ValueError()
        return iteration_kwargs

    def _ensemble_tree(self, params):
        train_data, test_data = self._get_dataset()

        columns = train_data.columns

        remove_columns = ['id', 'score']
        features_columns = [column for column in columns if column not in remove_columns]

        train_labels = train_data['score']
        train_x = train_data[features_columns]
        test_x = test_data[features_columns]

        kfolder = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=self.seed)
        kfold = kfolder.split(train_x, train_labels)

        preds_list = list()
        oof = np.zeros(train_data.shape[0])
        for train_index, vali_index in kfold:
            k_x_train = train_x[train_index]
            k_y_train = train_labels.loc[train_index]
            k_x_vali = train_x[vali_index]
            k_y_vali = train_labels.loc[vali_index]

            gbm = self._get_gbm(params)
            gbm = gbm.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                          early_stopping_rounds=200, verbose=False)
            iteration_kwargs = self._get_iteration_kwargs(gbm)
            k_pred = gbm.predict(k_x_vali, **iteration_kwargs)
            oof[vali_index] = k_pred

            preds = gbm.predict(test_x, **iteration_kwargs)
            preds_list.append(preds)

        fold_mae_error = mean_absolute_error(train_labels, oof)
        print('{} fold mae error is {}'.format(self.mode, fold_mae_error))
        fold_score = 1 / (1 + fold_mae_error)
        print('fold score is {}'.format(fold_score))

        preds_columns = ['preds_{id}'.format(id=i) for i in range(self.n_fold)]
        preds_df = pd.DataFrame(data=preds_list)
        preds_df = preds_df.T
        preds_df.columns = preds_columns
        preds_list = list(preds_df.mean(axis=1))
        prediction = preds_list

        if self.save:
            sub_df = pd.DataFrame({'id': test_data['id'],
                                   'score': prediction})
            sub_df['score'] = sub_df['score'].apply(lambda item: int(round(item)))
            sub_df.to_csv('submittion.csv', index=False)

        return oof, prediction