#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import utils
import train
import pandas as pd
from process_dataframe import processed_df

train_dataset_path = 'data/train_dataset.csv'
pred_dataset_path = 'data/test_dataset.csv'
model_path = 'model/model.txt'


# read data
train_dataset = pd.read_csv(train_dataset_path)
pred_dataset  = pd.read_csv(pred_dataset_path)
train_label = train_dataset['信用分']
train_dataset = train_dataset.drop(columns=['用户编码', '信用分'])
pred_dataset  = pred_dataset.drop(columns=['用户编码'])


# handle data
'''
In  : print(train_dataset.columns)
Out : Index(['用户实名制是否通过核实', '用户年龄', '是否大学生客户', '是否黑名单客户', '是否4G不健康客户',
       '用户网龄（月）', '用户最近一次缴费距今时长（月）', '缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）',
       '用户账单当月总费用（元）', '用户当月账户余额（元）', '缴费用户当前是否欠费缴费', '用户话费敏感度', '当月通话交往圈人数',
       '是否经常逛商场的人', '近三个月月均商场出现次数', '当月是否逛过福州仓山万达', '当月是否到过福州山姆会员店', '当月是否看电影',
       '当月是否景点游览', '当月是否体育场馆消费', '当月网购类应用使用次数', '当月物流快递类应用使用次数',
       '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数', '当月飞机类应用使用次数', '当月火车类应用使用次数',
       '当月旅游资讯类应用使用次数'],dtype='object')
'''
train_dataset = processed_df(train_dataset)
pred_dataset  = processed_df(pred_dataset)

# parameters setting
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # GBDT算法为基础
    'objective': 'regression',  # 回归任务
    'metric': 'rmse',  # 评判指标
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
    'n_estimator':1444,
    'reg_alpha': 0.5,
    'reg_lambda': 0.08,
    'header': False  # 数据集是否带表头
    }
params2 = params
params2['seed'] = 89

# train model
pred1 = train.train_model(train_dataset, train_label, pred_dataset, params, en_amount=3)
pred2 = train.train_model(train_dataset, train_label, pred_dataset, params2, en_amount=3)

pred = (pred1 + pred2) / 2
pred_list = pred.tolist()
pred  = [int(round(score)) for score in pred_list]
utils.write_SubmitionFile(pred, 'data/submit.csv')

