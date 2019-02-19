#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import utils
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def train(train_dataset, train_label, valid_dataset, valid_label, num_round, params, model_path):
    # 训练集加载
    train_data = lgb.Dataset(train_dataset, label=train_label)
    train_data.save_binary('data/train.bin')

    # 验证集加载
    valid_data = lgb.Dataset(valid_dataset, label=valid_label, reference=train_data)
    valid_data.save_binary('data/valid.bin')

    # Train & save model as model.txt
    bst = lgb.train(params=params, train_set=train_data, num_boost_round=num_round, valid_sets=None)
    bst.save_model(model_path)


def make_predtion(dataset, model_path):
    '''
    :param dataset: 需要预测的数据
    :param model_path: 模型路径
    :return:
    '''

    bst = lgb.Booster(model_file=model_path)
    pred = bst.predict(dataset)

    return pred

def handle_data(train_dataset_path, pred_dataset_path):
    # data_array: shape = (50000,28)
    train_data_array = utils.build_data_array(train_dataset_path, tag='train')
    pred_data_array = utils.build_data_array(pred_dataset_path, tag='pred')
    pred_dataset = pred_data_array

    # 拆分数据集 & 分离data、label
    train_dataset, validation = train_test_split(train_data_array, test_size=0.2, random_state=21)
    train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.25, random_state=21) # 80% * 0.25 = 100% * 0.2

    train_dataset, train_label = utils.split_data_and_label(train_dataset)
    valid_dataset, valid_label = utils.split_data_and_label(validation)
    test_dataset, test_label = utils.split_data_and_label(test_dataset)

    return train_dataset, train_label, valid_dataset, valid_label, test_dataset, test_label, pred_dataset
