#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import pandas as pd
import lightgbm as lgb
import utils
from sklearn import metrics
from sklearn.model_selection import train_test_split


train_dataset_path = 'data/train_dataset.csv'

# read data
train_dataset = pd.read_csv(train_dataset_path)
train_label = train_dataset['信用分']
train_dataset = utils.processed_df(train_dataset)
train_dataset = train_dataset.drop(columns=['用户编码', '信用分'])


X = train_dataset
y = train_label


### 数据转换
print('数据转换')
lgb_train = lgb.Dataset(train_dataset, train_label, free_raw_data=False)


### 设置初始参数--不含交叉验证参数
print('设置参数')
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'regression_l1',
    'max_bin': 511,  # 大会有更准的效果,更慢的速度
    'learning_rate': 0.01,  # 学习率
    'num_leaves': 31,  # 大会更准,但可能过拟合
    'max_depth': 5,  # 小数据集下限制最大深度可防止过拟合,小于0表示无限制
    'feature_fraction': 0.5,  # 如果 feature_fraction 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征.
                              # 例如, 如果设置为 0.8, 将会在每棵树训练之前选择 80% 的特征. 可以处理过拟合
    'bagging_freq': 5,  # 防止过拟合
    'bagging_fraction': 0.6,  # 防止过拟合
    'min_data_in_leaf': 50,  # 防止过拟合
    'min_sum_hessian_in_leaf': 3.0,  # 防止过拟合
    'reg_alpha': 0,
    'reg_lambda': 5,
}

### 交叉验证(调参)
print('交叉验证')
min_regression_l1 = 100.0
best_params = {}

# 准确率
print("调参1：提高准确率")
for num_leaves in range(5, 100, 5):
    for max_depth in range(3, 8, 1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth

        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=1,
            nfold=5,
            metrics=['regression_l1'],
            early_stopping_rounds=10,
            verbose_eval=True
        )
        mean_regression_l1 = pd.Series(cv_results['l1-mean']).min()
        boost_rounds = pd.Series(cv_results['l1-mean']).idxmax()

        if mean_regression_l1 <= min_regression_l1:
            min_regression_l1 = mean_regression_l1
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth
if 'num_leaves' and 'max_depth' in best_params.keys():
    params['num_leaves'] = best_params['num_leaves']
    params['max_depth'] = best_params['max_depth']

# 过拟合
print("调参2：降低过拟合")
min_regression_l1 = 100.0
for max_bin in range(100, 520, 10):
    for min_data_in_leaf in range(1, 101, 10):
        params['max_bin'] = max_bin
        params['min_data_in_leaf'] = min_data_in_leaf

        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=1,
            nfold=5,
            metrics=['regression_l1'],
            early_stopping_rounds=10,
            verbose_eval=True
        )

        mean_regression_l1 = pd.Series(cv_results['l1-mean']).min()
        boost_rounds = pd.Series(cv_results['l1-mean']).idxmax()

        if mean_regression_l1 <= min_regression_l1:
            min_regression_l1 = mean_regression_l1
            best_params['max_bin'] = max_bin
            best_params['min_data_in_leaf'] = min_data_in_leaf
if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
    params['min_data_in_leaf'] = best_params['min_data_in_leaf']
    params['max_bin'] = best_params['max_bin']

print("调参3：降低过拟合")
min_regression_l1 = 100.0
for feature_fraction in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for bagging_fraction in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_freq in range(0, 50, 5):
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = bagging_freq

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['regression_l1'],
                early_stopping_rounds=10,
                verbose_eval=True
            )

            mean_regression_l1 = pd.Series(cv_results['l1-mean']).min()
            boost_rounds = pd.Series(cv_results['l1-mean']).idxmax()

            if mean_regression_l1 <= min_regression_l1:
                min_regression_l1 = mean_regression_l1
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq

if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
    params['feature_fraction'] = best_params['feature_fraction']
    params['bagging_fraction'] = best_params['bagging_fraction']
    params['bagging_freq'] = best_params['bagging_freq']

print("调参4：降低过拟合")
min_regression_l1 = 100.0
for lambda_l1 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    for lambda_l2 in range(11):
        params['lambda_l1'] = lambda_l1
        params['lambda_l2'] = lambda_l2
        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=1,
            nfold=5,
            metrics=['regression_l1'],
            early_stopping_rounds=10,
            verbose_eval=True
        )

        mean_regression_l1 = pd.Series(cv_results['l1-mean']).min()
        boost_rounds = pd.Series(cv_results['l1-mean']).idxmax()

        if mean_regression_l1 <= min_regression_l1:
            min_regression_l1 = mean_regression_l1
            best_params['lambda_l1'] = lambda_l1
            best_params['lambda_l2'] = lambda_l2
if 'lambda_l1' and 'lambda_l2' in best_params.keys():
    params['lambda_l1'] = best_params['lambda_l1']
    params['lambda_l2'] = best_params['lambda_l2']

print("调参5：降低过拟合2")
min_regression_l1 = 100.0
for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    params['min_split_gain'] = min_split_gain

    cv_results = lgb.cv(
        params,
        lgb_train,
        seed=1,
        nfold=5,
        metrics=['regression_l1'],
        early_stopping_rounds=10,
        verbose_eval=True
    )

    mean_regression_l1 = pd.Series(cv_results['l1-mean']).min()
    boost_rounds = pd.Series(cv_results['l1-mean']).idxmax()

    if mean_regression_l1 <= min_regression_l1:
        min_regression_l1 = mean_regression_l1

        best_params['min_split_gain'] = min_split_gain
if 'min_split_gain' in best_params.keys():
    params['min_split_gain'] = best_params['min_split_gain']

print(best_params)
