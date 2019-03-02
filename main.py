#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import utils
import train
import time
import pandas as pd


train_dataset_path = 'data/train_dataset.csv'
pred_dataset_path = 'data/test_dataset.csv'


# read data
train_dataset = pd.read_csv(train_dataset_path)
pred_dataset  = pd.read_csv(pred_dataset_path)
train_label = train_dataset['信用分']
submition = pred_dataset[['用户编码']]
train_dataset = train_dataset.drop(columns=['用户编码', '信用分'])
pred_dataset  = pred_dataset.drop(columns=['用户编码'])


# 特征工程
train_dataset = utils.processed_df(train_dataset)
pred_dataset  = utils.processed_df(pred_dataset)


# Parameters setting
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mae',
    'learning_rate': 0.01,
    'min_child_samples': 46,
    'min_child_weight': 0.01,
    'bagging_freq': 2,
    'num_leaves': 86,
    'max_depth': 7,
    'bagging_fraction': 0.6,
    'feature_fraction': 0.4,
    'lambda_l1': 0.01,
    'lambda_l2': 0.55,
    'max_bin': 383,
    'verbose': -1,
    'bagging_seed': 4590
    }

# train and predict
pred, valid_score = train.train(train_dataset, train_label, pred_dataset, params)

# 将预测结果四舍五入，转化为要求格式
pred_list = pred.tolist()
pred_format = [int(round(each)) for each in pred_list]
score = 1 / (1 + valid_score)
print('\n', 'This prediction gets cv score for valid is : {}'.format(score))

# 将结果按赛制要求写入文件
submition['score'] = pred_format
submition.columns = ['id','score']
submition.to_csv('data/submition.csv', header=True, index=False)

# 将训练参数、模型保存路径和模型得分写入日志文件
utils.write_log(save_path='training_log.txt', Time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                TrainMethod='main1', Params=params, Score=score)

