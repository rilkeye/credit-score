#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import time
import utils
import train
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

train_dataset_path = 'data/train_dataset.csv'
pred_dataset_path = 'data/test_dataset.csv'
model_path = 'model/model.txt'

# read data
train_dataset = pd.read_csv(train_dataset_path)
train_label = train_dataset[['信用分']]
pred_dataset  = pd.read_csv(pred_dataset_path)

submition = pred_dataset[['用户编码']]
train_dataset = train_dataset.drop(columns=['用户编码'])
pred_dataset  = pred_dataset.drop(columns=['用户编码'])

train_dataset = utils.processed_df(train_dataset)
pred_dataset  = utils.processed_df(pred_dataset)

# params setting
lgb_params = {
            'boosting_type': 'gbdt',  # GBDT算法为基础
            'objective': 'regression',  # 回归任务
            'metric': 'regression_l1',  # 评判指标 regression_l1's alias: mae
            'max_bin': 255,  # 大会有更准的效果,更慢的速度
            'learning_rate': 0.01,  # 学习率
            'num_leaves': 40,  # 大会更准,但可能过拟合
            'max_depth': 8,  # 小数据集下限制最大深度可防止过拟合,小于0表示无限制
            'feature_fraction': 0.61,  # 如果 feature_fraction 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征.
                              # 例如, 如果设置为 0.8, 将会在每棵树训练之前选择 80% 的特征. 可以处理过拟合
            'bagging_freq': 5,  # 防止过拟合
            'bagging_fraction': 0.75,  # 防止过拟合
            'min_data_in_leaf': 21,  # 防止过拟合
            'min_sum_hessian_in_leaf': 3.0,  # 防止过拟合
            'lambda_l1': 0.5,
            'lambda_l2': 0.08,
             }

xgb_params = {'learning_rate': 0.003, 'n_estimators': 8000, 'max_depth': 6, 'min_child_weight': 10, 'seed': 0,
              'subsample': 0.6, 'colsample_bytree': 0.5, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 5, 'n_jobs': 20}

NFOLDS = 10
kfold = RepeatedKFold(n_splits=NFOLDS, n_repeats=NFOLDS, random_state=2019)
kf = kfold.split(train_dataset, train_label)

lgb_val_pred_list = list()
xgb_val_pred_list = list()
lgb_val_label_list = list()
xgb_val_label_list = list()
test_pred_df = submition

for i, (train_index, valid_index) in enumerate(kf):
    print('Fold {} is training...'.format(i))
    t0 = time.time()
    k_x_train, k_x_valid = train_dataset.iloc[train_index, :], train_dataset.iloc[valid_index, :]
    k_y_train, k_y_valid = train_label.iloc[train_index], train_label.iloc[valid_index]

    lgb_gbm = train.get_gbm(lgb_params, mode='lgb')
    lgb_gbm = lgb_gbm.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_valid, k_y_valid)],
                          early_stopping_rounds=100, verbose=False)
    lgb_val_pred = lgb_gbm.predict(k_x_valid)
    lgb_pred = lgb_gbm.predict(pred_dataset)
    print(lgb_val_pred, lgb_pred)
    lgb_val_pred_list.append(lgb_val_pred)
    test_pred_df['fold{}_lgb'.format(i)] = lgb_pred

    lgb_mae = mean_absolute_error(lgb_val_pred, k_y_valid)
    lgb_score = 1 / (1 + lgb_mae)
    print('Fold {} lgb model had trained, with valid_score : {}'.format(i, lgb_score))

    xgb_gbm = train.get_gbm(xgb_params, mode='xgb')
    xgb_gbm = xgb_gbm.fit(k_x_train, k_y_train, early_stopping_rounds=100, verbose=False)
    xgb_val_pred = xgb_gbm.predict(k_x_valid)
    xgb_pred = xgb_gbm.predict(pred_dataset)

    xgb_val_pred_list.append(xgb_val_pred)
    test_pred_df['fold{}_xgb'.format(i)] = xgb_pred

    xgb_mae = mean_absolute_error(xgb_val_pred, k_y_valid)
    xgb_score = 1 / (1 + xgb_mae)
    print('Fold {} xgb model had trained, with valid_score : {}'.format(i, lgb_score))

    usage_time = time.time() - t0
    print('Fold {} had trained, using {} seconds.'.format(i, usage_time))

lgb_train_stack = np.vstack(lgb_val_pred_list)
# lgb_test_stack =