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
params = {
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

train_data1, train_data2 = train_test_split(train_dataset, test_size=0.2, random_state=21)
train_data1, train_data2 = train_data1.reset_index(), train_data2.reset_index()
# train_data1, train_data2 = train_dataset, train_dataset
train_label1 = train_data1['信用分']
train_label2 = train_data2['信用分']
train_data1 = train_data1.drop(columns=['信用分', 'index'])
train_data2 = train_data2.drop(columns=['信用分', 'index'])
print(train_data2, pred_dataset)

train_pred1, pred1, valid_score1 = train.train3_lgb(train_data1, train_label1, train_data2, pred_dataset, params)
train_pred2, pred2, valid_score2 = train.train3_xgb(train_data1, train_label1, train_data2, pred_dataset)
print("lgb1 score is: ", 1 / (1 + valid_score1))
print("xgb2 score is: ", 1 / (1 + valid_score2))
train_pred1, train_pred2 = pd.DataFrame(train_pred1), pd.DataFrame(train_pred2)
print(pred1.shape)
pred1, pred2 = pred1.reshape(-1, 1), pred2.reshape(-1, 1)
print(pred1.shape)

linear1 = LinearRegression().fit(train_pred1, train_label2)
pred3 = linear1.predict(pred1)
# print("linear1 score is: ", 1 / (1 + metrics.mean_absolute_error()))

linear2 = LinearRegression().fit(train_pred2, train_label2)
pred4 = linear2.predict(pred2)



# 两次预测结果求平均值
pred = (pred3 + pred4 ) / 2
valid_score = (valid_score1 + valid_score2) / 2
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
utils.write_log(save_path='training log.txt', Time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                TrainMethod='main2', Params=params, Score=score)