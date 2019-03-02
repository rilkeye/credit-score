#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke


import time
import utils
import train
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

train_dataset_path = 'data/train_dataset.csv'
pred_dataset_path = 'data/test_dataset.csv'
model_path = 'model/model.txt'


# read data
train_dataset = pd.read_csv(train_dataset_path)
pred_dataset  = pd.read_csv(pred_dataset_path)
# train_label = train_dataset['信用分']
submition = pred_dataset[['用户编码']]
train_dataset = train_dataset.drop(columns=['用户编码'])
pred_dataset  = pred_dataset.drop(columns=['用户编码'])

train_dataset = utils.processed_df(train_dataset)
pred_dataset  = utils.processed_df(pred_dataset)

# lgb params
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': 'mae',
        'learning_rate': 0.01,
        'min_child_samples': 46,
        'min_child_weight': 0.01,
        'bagging_freq': 2,
        'num_leaves': 90,
        'max_depth': 7,
        'bagging_fraction': 0.6,
        'feature_fraction': 0.4,
        'lambda_l1': 0.01,
        'lambda_l2': 0.55,
        'max_bin': 383,
        'verbose': -1,
        'bagging_seed': 4590
    }
xgb_params = {'learning_rate': 0.003, 'n_estimators': 8000, 'max_depth': 6, 'min_child_weight': 10, 'seed': 0,
              'subsample': 0.6, 'colsample_bytree': 0.5, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 5, 'n_jobs': 20}


train_data1, train_data2 = train_test_split(train_dataset, test_size=0.2, random_state=21)
train_data1, train_data2 = train_data1.reset_index(), train_data2.reset_index()
# train_data1, train_data2 = train_dataset, train_dataset
train_label1 = train_data1['信用分']
train_label2 = train_data2['信用分']
train_data1 = train_data1.drop(columns=['信用分', 'index'])
train_data2 = train_data2.drop(columns=['信用分', 'index'])

train_pred1, pred1, valid_score1 = train.train3_lgb(train_data1, train_label1, train_data2, pred_dataset, lgb_params)
train_pred2, pred2, valid_score2 = train.train3_xgb(train_data1, train_label1, train_data2, pred_dataset, xgb_params)
print("lgb1 score is: ", 1 / (1 + valid_score1))
print("xgb2 score is: ", 1 / (1 + valid_score2))

train_pred1, train_pred2 = pd.DataFrame(train_pred1), pd.DataFrame(train_pred2)
train_label2 = pd.DataFrame(train_label2)
pred1, pred2 = pred1.reshape(-1, 1), pred2.reshape(-1, 1)

pred3, valid_score3 = train.train3_stacking(train_pred1, train_label2, pred1)
pred4, valid_score4 = train.train3_stacking(train_pred2, train_label2, pred2)

# 两次预测结果求平均值
pred = (pred3 + pred4 ) / 2
valid_score = (valid_score3 + valid_score4) / 2
score = 1 / (1 + valid_score)
print("Final score is: ", 1 / (1 + valid_score))


# 将预测结果四舍五入，转化为要求格式
pred_list = pred.tolist()
pred_format = [int(round(each)) for each in pred_list]
# print(pred_format[:100])
print('\n', 'This prediction gets cv score for valid is : {}'.format(score))

# 将结果按赛制要求写入文件
submition['score'] = pred_format
submition.columns = ['id','score']
submition.to_csv('data/submition2.csv', header=True, index=False)

# 将训练参数、模型保存路径和模型得分写入日志文件
utils.write_log(save_path='training_log.txt', Time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                TrainMethod='main3', LGB_Params=lgb_params, XGB_Params=xgb_params, Score=score)