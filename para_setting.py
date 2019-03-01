#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import numpy as np
import pandas as pd
import lightgbm as lgb
import utils
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit

'''
# We had run this .py file getting the best parameters:
{
    'n_estimators': 1655,
    'max_depth': 7, 
    'num_leaves': 66,
    'min_data_in_leaf': 20,
    'min_sum_hessian_in_leaf': 0.001,
    'feature_fraction': 0.5, 
    'bagging_fraction': 0.01,
    'lambda_l1': 0.01, 
    'lambda_l2': 0.5,
    'max_bin': 381
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
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'n_estimators': 10000,  # base :
        'metric': 'mae',
        'learning_rate': 0.01,  # base : 0.01~0.02
        'min_child_samples': 46,  # base : 30~50
        'min_child_weight': 0.01, # base :
        'bagging_freq': 2,
        'num_leaves': 40,
        'max_depth': 7,
        'bagging_fraction': 0.6,
        'feature_fraction': 0.8,
        'lambda_l1': 0,
        'lambda_l2': 5,
        'verbose': -1,
        'bagging_seed': 4590
    }

param_range = range(1000, 10000, 1000)
para_name = 'n_estimators'

dtrain = lgb.Dataset(train_dataset, train_label)
gbm = lgb.LGBMRegressor()
# train_size, train_scores, test_scores = learning_curve(gbm, train_dataset, train_label,
#                                                        train_sizes=np.linspace(.05, 1., 15), cv=10)
train_scores, test_scores = validation_curve(gbm, train_dataset, train_label,
                                             param_name=para_name, param_range=param_range, cv=10)
train_scores_mean = np.mean(train_scores, axis=1)
train_score_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.fill_between(param_range, train_scores_mean - train_score_std, train_scores_mean + train_score_std, alpha=0.1)
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=.1, color='g')
plt.axhline(np.mean(test_scores_mean), color='b')
plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.ylabel('ROC_AUC')
plt.xlabel(para_name)
plt.legend(loc='best')

# plt.semilogx(param_range, train_scores_mean, label='training score', color='r')
# plt.semilogx(param_range, test_scores_mean, label='cross-validation score', color='g')
plt.show()

'''
# --------------------------------------------------------------
dtrain = lgb.Dataset(train_dataset, train_label)
cv_results = lgb.cv(
    params, dtrain, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='regression_l1',
    early_stopping_rounds=100, verbose_eval=50, show_stdv=True, seed=0)
print(cv_results) 
print('best n_estimators:', len(cv_results['l1-mean']))
print('best cv score:', cv_results['l1-mean'][-1])
# -------------------------------------------------------------
# best n_estimators: 2561
'''

'''
# -------------------------------------------------------------
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=32,
                              learning_rate=0.01, n_estimators=1655, max_depth=5,
                              metric='regression_l1', bagging_fraction=0.6, feature_fraction=0.5)
params_test1={
    'max_depth': [5], # range(4,8,1),
    'num_leaves': [20]# range(30, 120, 10)
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
    'min_data_in_leaf': [19, 20, 21, 22],
    'min_sum_hessian_in_leaf':[0.001, 3.0]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=66,
                              learning_rate=0.01, n_estimators=1655, max_depth=7,
                              metric='regression_l1', bagging_fraction = 0.6, feature_fraction = 0.5)
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch3.fit(train_dataset, train_label)
print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)
# -------------------------------------------------------------
# 'min_data_in_leaf': 20, 'min_sum_hessian_in_leaf': 0.001
# best_score :-14.710041112943099
'''


'''
# -------------------------------------------------------------
params_test4={
    'feature_fraction': [0.5],
    'bagging_fraction':[0.01, 0.03, 0.04]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=66,
                              learning_rate=0.01, n_estimators=1655, max_depth=7, min_data_in_leaf=20, min_sum_hessian_in_leaf=0.001,
                              metric='regression_l1', bagging_fraction = 0.6, feature_fraction = 0.5)
gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch4.fit(train_dataset, train_label)
print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)
# -------------------------------------------------------------
# 'feature_fraction': 0.5, 'bagging_fraction': 0.01
# best_score :-14.710041112943099
'''

'''
params_test5={
    'lambda_l1': [0.01, 0.1, 0.5],
    'lambda_l2': [0.45, 0.5, 0.55]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=66,
                              learning_rate=0.01, n_estimators=1655, max_depth=7,min_data_in_leaf=19, min_sum_hessian_in_leaf=0.001,
                              metric='regression_l1', bagging_fraction = 0.01, feature_fraction = 0.5)
gsearch5 = GridSearchCV(estimator=model_lgb, param_grid=params_test5, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch5.fit(train_dataset, train_label)
print(gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_)
# -------------------------------------------------------------
# 'lambda_l1': 0.01, 'lambda_l2': 0.5
# best_score : -14.710015134819564
'''


# set new params
params = {
              'n_estimators': 1655,
              'max_depth': 7,
              'num_leaves': 66,
              'min_data_in_leaf': 20,
              'min_sum_hessian_in_leaf': 0.001,
              'feature_fraction': 0.5,
              'bagging_fraction': 0.01,
              'lambda_l1': 0.01,
              'lambda_l2': 0.5
              }

'''
# -------------------------------------------------------------
data_train = lgb.Dataset(train_dataset, train_label, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='regression_l1',
    early_stopping_rounds=100, verbose_eval=1, show_stdv=True)

print('best n_estimators:', len(cv_results['l1-mean']))
print('best cv score:', cv_results['l1-mean'][-1])
# best n_estimators: 152
# best cv score: 14.855494695121404
# -------------------------------------------------------------
'''

'''
# -------------------------------------------------------------
params_test6={
    'max_bin':range(375, 386, 1)
}
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=66,
                              learning_rate=0.01, n_estimators=1655, max_depth=7,min_data_in_leaf=20, min_sum_hessian_in_leaf=0.001,
                              metric='regression_l1', bagging_fraction=0.01, feature_fraction=0.5, lambda_l1=0.01, lambda_l2=0.5)
gsearch6 = GridSearchCV(estimator=model_lgb, param_grid=params_test6, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch6.fit(train_dataset, train_label)
print(gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_)
# 'max_bin': 381
# best cv score:-14.714731203914459
# -------------------------------------------------------------
'''