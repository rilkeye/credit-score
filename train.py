#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import utils
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def train(train_data, train_label, pred_data, params):
    t0 = utils.timer()
    x_train, x_validate = train_test_split(train_data, test_size=0.2, random_state=21)
    label_train, label_validate = train_test_split(train_label, test_size=0.2, random_state=21)

    print( 'training...')

    bst = lgb.LGBMRegressor(**params)
    bst.fit(x_train, label_train, eval_metric=mean_absolute_error)

    # 预测
    pred = bst.predict(pred_data)
    valid_score = mean_absolute_error(label_validate, bst.predict(x_validate))

    usage_time = utils.timer() - t0
    print('usage time for train3_lgb is : {} seconds'.format(usage_time))
    return pred, valid_score


def train2_lgb(train_data, train_label, pred_data, params, en_amount):
    t0 = utils.timer()
    pred_all = 0
    valid_score = 0
    for seed in range(en_amount):
        NFOLDS = 5
        kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
        kf = kfold.split(train_data, train_label)
        count = 0
        pred = np.zeros(pred_data.shape[0])
        valid_best_all = 0

        for i, (train_fold, validate) in enumerate(kf):
            print('Fold: ', i, ' training...')
            # K_Fold 分割数据集，验证集
            x_train, x_validate, label_train, label_validate = \
                train_data.iloc[train_fold, :], train_data.iloc[validate, :], train_label[train_fold], train_label[validate]

            bst = lgb.LGBMRegressor(**params)
            bst.fit(x_train, label_train, eval_metric=mean_absolute_error)

            pred += bst.predict(pred_data)
            valid_best_all += mean_absolute_error(label_validate, bst.predict(x_validate))

            count += 1
        pred /= NFOLDS
        valid_best_all /= NFOLDS
        print('cv score for valid is: ', 1 / (1 + valid_best_all))
        pred_all += pred
        valid_score += valid_best_all

    pred_all /= en_amount
    valid_score /= en_amount
    usage_time = utils.timer() - t0
    print('usage time for train3_lgb is : {} seconds'.format(usage_time))
    return pred_all, valid_score

def train2_xgb(train_data, train_label, pred_data, params):
    t0 = utils.timer()
    NFOLDS = 5
    kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2019)
    kf = kfold.split(train_data, train_label)

    # init var
    pred = np.zeros(pred_data.shape[0])
    score = 0

    # model=None
    for i, (train_index, val_index) in enumerate(kf):
        print('fold: ', i, ' training...')
        x_train, x_validate, label_train, label_validata = train_data.iloc[train_index, :], train_data.iloc[val_index, :], \
                                         train_label[train_index], train_label[val_index]

        # train
        model = xgb.XGBRegressor(**params)
        model.fit(x_train, label_train, eval_metric=mean_absolute_error)

        # predict & score calculate
        pred += model.predict(pred_data)
        score += mean_absolute_error(label_validata, model.predict(x_validate))

    pred = pred / NFOLDS
    score /= NFOLDS
    usage_time = utils.timer() - t0
    print('usage time for train3_lgb is : {} seconds'.format(usage_time))
    return pred, score

def train3_lgb(train_data, train_label, train_data2, pred_data, params):
    t0 = utils.timer()
    NFOLDS = 5
    kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2019)
    kf = kfold.split(train_data, train_label)
    count = 0
    train2_pred = np.zeros(train_data2.shape[0])
    pred = np.zeros(pred_data.shape[0])
    valid_best_all = 0
    for i, (train_fold, validate) in enumerate(kf):
        print('Fold: ', i, ' training...')
        # K_Fold 分割数据集，验证集
        x_train, x_validate, label_train, label_validate = \
            train_data.iloc[train_fold, :], train_data.iloc[validate, :], train_label[train_fold], train_label[validate]

        bst = lgb.LGBMRegressor(**params)
        bst.fit(x_train, label_train, eval_metric=mean_absolute_error)

        train2_pred += bst.predict(train_data2)
        pred += bst.predict(pred_data)
        # bst.best_score example : {'valid_0': {'l1': 14.744103789371326}}
        valid_best_all += mean_absolute_error(label_validate, bst.predict(x_validate))
        count += 1
    train2_pred /= NFOLDS
    pred /= NFOLDS
    valid_best_all /= NFOLDS
    print('cv score for valid is: ', 1 / (1 + valid_best_all))
    usage_time = utils.timer() - t0
    print('usage time for train3_lgb is : {} seconds'.format(usage_time))
    return train2_pred, pred, valid_best_all


def train3_xgb(train_data, train_label, train2_data, pred_data):
    t0 = utils.timer()
    NFOLDS = 5
    kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2019)
    kf = kfold.split(train_data, train_label)

    # init var
    train2_pred = np.zeros(train2_data.shape[0])
    pred = np.zeros(pred_data.shape[0])
    score = 0

    # model=None
    for i, (train_index, val_index) in enumerate(kf):
        print('fold: ', i, ' training...')
        x_train, x_validate, label_train, label_validata = train_data.iloc[train_index, :], train_data.iloc[val_index, :], \
                                         train_label[train_index], train_label[val_index]
        # train
        params = {'learning_rate': 0.003, 'n_estimators': 8000, 'max_depth': 6, 'min_child_weight': 10, 'seed': 0,
                  'subsample': 0.6, 'colsample_bytree': 0.5, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 5, 'n_jobs': 20}
        model = xgb.XGBRegressor(**params)
        model.fit(x_train, label_train, eval_metric=mean_absolute_error)

        # predict & score calculate
        train2_pred += model.predict(train2_data)
        pred += model.predict(pred_data)
        score += mean_absolute_error(label_validata, model.predict(x_validate))

    train2_pred /= NFOLDS
    pred = pred / NFOLDS
    score /= NFOLDS
    usage_time = utils.timer() - t0
    print('usage time for train3_xgb is : {} seconds'.format(usage_time))
    return train2_pred, pred, score

def train3_stacking(train_data, train_label, pred_data):
    t0 = utils.timer()
    NFOLD = 5
    kfold = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=2019)
    kf = kfold.split(train_data, train_label)

    # init var
    stacking_pred = np.zeros(pred_data.shape[0])
    valid_score = 0

    for i, (train_index, val_index) in enumerate(kf):
        print('fold : {} training...'.format(i))
        k_x_train = train_data.iloc[train_index]
        k_y_train = train_label.iloc[train_index]
        k_x_valid = train_data.iloc[val_index]
        k_y_valid = train_label.iloc[val_index]

        gbm = BayesianRidge(normalize=True)
        gbm.fit(k_x_train, k_y_train)

        k_pred = gbm.predict(k_x_valid)
        valid_score += mean_absolute_error(k_pred, k_y_valid)

        preds = gbm.predict(pred_data)
        stacking_pred += preds

    stacking_pred /= NFOLD
    valid_score /= NFOLD
    print('stacking fold mae error is {}'.format(valid_score))
    fold_score = 1 / (1 + valid_score)
    print('fold score is {} '.format(fold_score))

    usage_time = utils.timer() - t0
    print('usage time for train3_stacking is : {} seconds'.format(usage_time))
    return stacking_pred, valid_score

def get_gbm(params, mode):
    if mode == 'lgb':
        gbm = lgb.LGBMRegressor(**params)
    elif mode == 'xgb':
        gbm = xgb.XGBRegressor(**params)
    else:
        raise ValueError()
    return gbm

