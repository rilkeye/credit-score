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
model_path = 'model/model.txt'

# read data
train_dataset = pd.read_csv(train_dataset_path)
pred_dataset  = pd.read_csv(pred_dataset_path)
train_dataset = train_dataset.drop(columns=['用户编码'])
pred_dataset = pred_dataset.drop(columns=['用户编码'])

# 特征工程
train_dataset = processed_df(train_dataset)
pred_data  = processed_df(pred_dataset)

# 训练集、验证集、测试集分割
train_data, train_label, valid_data, valid_label, test_data, test_label = train.handle_data(train_dataset)


# Parameters setting
num_round = 1444
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
train.train(train_data, train_label, valid_data, valid_label, num_round, params, model_path)

# 用测试集进行测试
test_pred = train.make_predtion(test_data, model_path)
score = utils.give_a_mark(test_pred, test_label) # 为当前载入模型评分
print("This Model gets score {}".format(score))

# 预测结果
result_pred = train.make_predtion(pred_data, model_path)
format_result_pred = [int(round(score)) for score in result_pred] # 将result_pred中的float型四舍五入

# 将结果按赛制要求写入文件
utils.write_SubmitionFile(format_result_pred, pred_dataset_path)
# print(format_result_pred[:100])


# 将训练参数、模型保存路径和模型得分写入日志文件
utils.write_log(save_path='training log.txt', Time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                Num_round=num_round, Params=params, Model_path=model_path, Score=score)