#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import pandas as pd
import lightgbm as lgb
import utils
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

'''
# We had run this .py file getting the best parameters:
{
    'max_bin':200,
    'n_estimators': 1655,
    'max_depth': 7, 
    'num_leaves': 66,
    'min_data_in_leaf': 19, 
    'min_sum_hessian_in_leaf': 0.001,
    'feature_fraction': 0.4, 
    'bagging_fraction': 0.4,
}

'''

train_dataset_path = 'data/train_dataset.csv'

# read data
train_dataset = pd.read_csv(train_dataset_path)
train_label = train_dataset['信用分']
train_dataset = utils.processed_df(train_dataset)
train_dataset = train_dataset.drop(columns=['用户编码', '信用分'])

# old params
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'regression_l1',
    'max_bin': 511,  # 大会有更准的效果,更慢的速度
    'learning_rate': 0.01,  # 学习率
    'num_leaves': 32,  # 大会更准,但可能过拟合
    'max_depth': 5,  # 小数据集下限制最大深度可防止过拟合,小于0表示无限制
    'feature_fraction': 0.5,  # 如果 feature_fraction 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征.
                              # 例如, 如果设置为 0.8, 将会在每棵树训练之前选择 80% 的特征. 可以处理过拟合
    'bagging_freq': 5,  # 防止过拟合
    'bagging_fraction': 0.6,  # 防止过拟合
    'min_data_in_leaf': 50,  # 防止过拟合
    'min_sum_hessian_in_leaf': 3.0,  # 防止过拟合
    'n_estimator': 1444,
    'reg_alpha': 0,
    'reg_lambda': 5,
    }

'''
# --------------------------------------------------------------
dtrain = lgb.Dataset(train_dataset, train_label)
cv_results = lgb.cv(
    params, dtrain, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='regression_l1',
    early_stopping_rounds=10, verbose_eval=50, show_stdv=True, seed=0)
print('best n_estimators:', len(cv_results['l1-mean']))
print('best cv score:', cv_results['l1-mean'][-1])
# -------------------------------------------------------------
# best n_estimators: 1655
'''

'''
# -------------------------------------------------------------
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=32,
                              learning_rate=0.01, n_estimators=1655, max_depth=5,
                              metric='regression_l1', bagging_fraction=0.6, feature_fraction=0.5)
params_test1={
    'max_depth': range(4,8,1),
    'num_leaves':range(30, 120, 10)
}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch1.fit(train_dataset, train_label)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
# -------------------------------------------------------------
# 'max_depth': 7, 'num_leaves': 70
# best_score : -14.713925782407046
'''

'''
# -------------------------------------------------------------
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=32,
                              learning_rate=0.01, n_estimators=1655, max_depth=5,
                              metric='regression_l1', bagging_fraction=0.6, feature_fraction=0.5)
params_test2={
    'max_depth': range(6,9,1),
    'num_leaves':range(65, 76)
}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test2, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch1.fit(train_dataset, train_label)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
# -------------------------------------------------------------
# 'max_depth': 7, 'num_leaves': 66
# best_score : -14.710041112943099
'''

'''
# -------------------------------------------------------------
params_test3={
    'min_data_in_leaf': [17, 18, 19],
    'min_sum_hessian_in_leaf':[0.001]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=66,
                              learning_rate=0.01, n_estimators=1655, max_depth=7,
                              metric='regression_l1', bagging_fraction = 0.6, feature_fraction = 0.5)
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch3.fit(train_dataset, train_label)
print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)
# -------------------------------------------------------------
# 'min_data_in_leaf': 19, 'min_sum_hessian_in_leaf': 0.001
# best_score :-366.12615009824333
'''


'''
# -------------------------------------------------------------
params_test4={
    'feature_fraction': [0.4, 0.5, 0.6, 0.7],
    'bagging_fraction':[0.4, 0.5, 0.6, 0.7]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=66,
                              learning_rate=0.01, n_estimators=1655, max_depth=7, min_data_in_leaf=19, min_sum_hessian_in_leaf=0.001,
                              metric='regression_l1', bagging_fraction = 0.6, feature_fraction = 0.5)
gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch4.fit(train_dataset, train_label)
print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)
# -------------------------------------------------------------
# 'feature_fraction': 0.4, 'bagging_fraction': 0.4
# best_score :-366.08896355990686
'''

'''
params_test5={
    'lambda_l1': [0, 0.001, 0.01, 0.1, 0.5],
    'lambda_l2': [0, 0.3, 0.5, 5]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=66,
                              learning_rate=0.01, n_estimators=1655, max_depth=7,min_data_in_leaf=19, min_sum_hessian_in_leaf=0.001,
                              metric='regression_l1', bagging_fraction = 0.4, feature_fraction = 0.4)
gsearch5 = GridSearchCV(estimator=model_lgb, param_grid=params_test5, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch5.fit(train_dataset, train_label)
print(gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_)
# -------------------------------------------------------------
# 'lambda_l1': 0.1, 'lambda_l2': 0.5
# best_score :-365.8680363591441
'''


# set new params
params = {
              'best n_estimators': 1655,
              'max_depth': 7,
              'num_leaves': 66,
              'min_data_in_leaf': 19,
              'min_sum_hessian_in_leaf': 0.001,
              'feature_fraction': 0.4,
              'bagging_fraction': 0.4,
              'lambda_l1': 0.1,
              'lambda_l2': 0.5
              }

'''
# -------------------------------------------------------------
data_train = lgb.Dataset(train_dataset, train_label, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='regression_l1',
    early_stopping_rounds=50, verbose_eval=100, show_stdv=True)

print('best n_estimators:', len(cv_results['l1-mean']))
print('best cv score:', cv_results['l1-mean'][-1])
# best n_estimators: 205
# best cv score: 14.819445741071798
# -------------------------------------------------------------
'''

'''
# -------------------------------------------------------------
params_test6={
    'max_bin':[100, 200, 300, 400, 510, 520]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=66,
                              learning_rate=0.01, n_estimators=1655, max_depth=7,min_data_in_leaf=19, min_sum_hessian_in_leaf=0.001,
                              metric='regression_l1', bagging_fraction = 0.4, feature_fraction = 0.4)
gsearch6 = GridSearchCV(estimator=model_lgb, param_grid=params_test6, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch6.fit(train_dataset, train_label)
print(gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_)
# 'max_bin': 200
# best cv score: -365.85004009169745
# -------------------------------------------------------------
'''