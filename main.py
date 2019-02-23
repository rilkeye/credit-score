#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import utils
import train
import time
import pandas as pd
from process_dataframe import processed_df


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
train_dataset = processed_df(train_dataset)
pred_dataset  = processed_df(pred_dataset)


# Parameters setting
params = {
    'task': 'train',
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
    'reg_alpha': 0.5,
    'reg_lambda': 0.08,
    'header': False  # 数据集是否带表头
    }

# train and predict
pred, valid_score = train.train(train_dataset, train_label, pred_dataset, params)

# 将预测结果四舍五入，转化为要求格式
pred_list = pred.tolist()
pred_format = [int(round(score)) for score in pred_list]
score = 1 / (1 + valid_score)
print('\n', 'This prediction gets cv score for valid is : {}'.format(score))

# 将结果按赛制要求写入文件
submition['score'] = pred_format
submition.columns = ['id','score']
submition.to_csv('data/submition.csv', header=True, index=False)

# 将训练参数、模型保存路径和模型得分写入日志文件
utils.write_log(save_path='training log.txt', Time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                TrainMethod='main1', Params=params, Score=score)
