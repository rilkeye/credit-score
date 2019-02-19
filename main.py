#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import utils
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import train_test_split

'''
载入train_dataset.csv、test_dataset.csv文件，构建二维数组
缺省train、test首列：“用户编码”；train末列：“信用分”
'''
train_dataset_path = 'data/train_dataset.csv'
test_dataset_path = 'data/test_dataset.csv'

# data_array: shape = (50000,28); score: shape = (50000,)
train_data_array = utils.build_data_array(train_dataset_path, tag='train')
test_data_array  = utils.build_data_array(test_dataset_path, tag='test')

# 拆分数据集 & 分离data、label
train_dataset, validation = train_test_split(train_data_array, test_size=0.2, random_state=21)
train_dataset, train_label = utils.split_data_and_label(train_dataset)
valid_dataset, valid_label = utils.split_data_and_label(validation)


# 训练集加载
train_data = lgb.Dataset(train_dataset, label=train_label)
train_data.save_binary('data/train.bin')

# 验证集加载
valid_data = lgb.Dataset(valid_dataset, label=valid_label, reference=train_data)
valid_data.save_binary('data/valid.bin')

# Parameters setting
num_round = 1000
params = {'objective':'regression', 'metric':'auc'}


# Train & save model as model.txt
bst = lgb.train(params=params, train_set=train_data, num_boost_round=num_round, valid_sets=None)
bst.save_model('model/model.txt')

# Load model and make prediction
bst = lgb.Booster(model_file='model/model.txt')
pred = bst.predict(test_data_array)
pred = [int(round(score)) for score in pred] # 将pred中的float型四舍五入

# 将结果按赛制要求写入文件
utils.write_SubmitionFile(pred, test_dataset_path)
print(pred[:100])