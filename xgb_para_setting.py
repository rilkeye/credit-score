#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import utils
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit


train_dataset_path = 'data/train_dataset.csv'

# read data
train_dataset = pd.read_csv(train_dataset_path)
train_label = train_dataset['信用分']
train_dataset = utils.processed_df(train_dataset)
train_dataset = train_dataset.drop(columns=['用户编码', '信用分'])

# old params
params = {'learning_rate': 0.003, 'n_estimators': 8000, 'max_depth': 6, 'min_child_weight': 10, 'seed': 0,
          'subsample': 0.6, 'colsample_bytree': 0.5, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 5, 'n_jobs': 20}
# max_delta_step : int ;

'''
param_range = np.linspace(0.001, 0.01, 10)
para_name = 'learning_rate'

dtrain = lgb.Dataset(train_dataset, train_label)
gbm = xgb.XGBRegressor()
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

'''
# --------------------------------------------------------------
dtrain = xgb.DMatrix(train_dataset, label=train_label)
cv_results = xgb.cv(
    params, dtrain, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='mae',
    early_stopping_rounds=100, verbose_eval=50, show_stdv=True, seed=0)
print(cv_results) 
print('best n_estimators:', len(cv_results['l1-mean']))
print('best cv score:', cv_results['l1-mean'][-1])
# -------------------------------------------------------------
# best n_estimators: 2538
# best_score : 14.695299624701544
'''

'''
# -------------------------------------------------------------
model_lgb = xgb.XGBRegressor(**params)

params_test1={
    'n_estimators': range(1000, 11000, 1000),  # range(4,9,1),
}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch1.fit(train_dataset, train_label)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
# -------------------------------------------------------------
'''

'''
# -------------------------------------------------------------
model_lgb = lgb.LGBMRegressor(**params)

params_test1={
    'max_depth': [7],  # range(4,9,1),
    'num_leaves': range(60, 120, 10)  # range(30, 120, 10)
}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch1.fit(train_dataset, train_label)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
# -------------------------------------------------------------
# 'max_depth': 7, 'num_leaves': 90
# best_score : -14.680602460767274
'''

'''
# -------------------------------------------------------------
model_lgb = lgb.LGBMRegressor(**params)
params_test2 = {
    'bagging_freq': range(1, 11)
}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test2, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch1.fit(train_dataset, train_label)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
# -------------------------------------------------------------
# 'bagging_freq': 2
# best_score : -14.656274158678205
'''

'''
# -------------------------------------------------------------
params_test3={
    'min_child_samples': [19, 20, 21, 22, 45, 46, 47],
    'min_child_weight': [0.01, 0.1, 0.5]
}
model_lgb = lgb.LGBMRegressor(**params)
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch3.fit(train_dataset, train_label)
print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)
# -------------------------------------------------------------
# 'min_data_in_leaf': 46, 'min_sum_hessian_in_leaf': 0.01
# best_score :-14.680602460767274
'''

'''
# -------------------------------------------------------------
params_test4={
    'feature_fraction': [0.4],
    'bagging_fraction': [0.5, 0.6, 0.7, 0.8]
}
model_lgb = lgb.LGBMRegressor(**params)
gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch4.fit(train_dataset, train_label)
print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)
# -------------------------------------------------------------
# 'feature_fraction': 0.4, 'bagging_fraction': 0.6
# best_score :-14.663373377652112
'''

'''
params_test5={
    'lambda_l1': [0.001, 0.01, 0.05],
    'lambda_l2': [0.55, 0.6]
}
model_lgb = lgb.LGBMRegressor(**params)
gsearch5 = GridSearchCV(estimator=model_lgb, param_grid=params_test5, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch5.fit(train_dataset, train_label)
print(gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_)
# -------------------------------------------------------------
# 'lambda_l1': 0.01, 'lambda_l2': 0.55
# best_score : -14.659112875263993
'''

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

'''
# -------------------------------------------------------------
data_train = lgb.Dataset(train_dataset, train_label, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='regression_l1',
    early_stopping_rounds=100, verbose_eval=1, show_stdv=True)

print('best n_estimators:', len(cv_results['l1-mean']))
print('best cv score:', cv_results['l1-mean'][-1])
# best n_estimators: 2806
# best cv score: 14.88031659859046
# -------------------------------------------------------------
'''

'''
# -------------------------------------------------------------
params_test6={
    'max_bin': range(375, 386, 1)
}
model_lgb = lgb.LGBMRegressor(**params)
gsearch6 = GridSearchCV(estimator=model_lgb, param_grid=params_test6, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)
gsearch6.fit(train_dataset, train_label)
print(gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_)
# 'max_bin': 383
# best cv score:-14.656274158678205
# -------------------------------------------------------------
'''
