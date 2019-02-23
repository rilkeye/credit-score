#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import utils
import lightgbm as lgb
import pandas as pd
import numpy as np
from process_dataframe import processed_df
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def train(train_dataset, train_label, valid_dataset, valid_label, num_round, params, model_path):
    # 训练集加载
    train_data = lgb.Dataset(train_dataset, label=train_label)
    train_data.save_binary('data/train.bin')

    # 验证集加载
    valid_data = lgb.Dataset(valid_dataset, label=valid_label, reference=train_data)
    valid_data.save_binary('data/valid.bin')

    # Train & save model as model.txt
    bst = lgb.train(params=params, train_set=train_data, num_boost_round=num_round, valid_sets=valid_data)
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
    train_data_df = utils.load_data(train_dataset_path, tag='train')
    train_data_df = processed_df(train_data_df) # 特征工程：处理dataframe

    pred_data_df = utils.load_data(pred_dataset_path, tag='pred')
    pred_data_df = processed_df(pred_data_df) # 特征工程：处理dataframe

    # 将“信用分”一列置于末列
    train_data_df_score = train_data_df['信用分']
    train_data_df = train_data_df.drop(columns=['信用分'])
    train_data_df.insert(train_data_df.shape[1], '信用分', train_data_df_score)

    # 将train_data 、 pred_data 转化为带入模型所需要的array格式
    train_data_array = train_data_df.values
    pred_data_array = pred_data_df.values
    pred_dataset = pred_data_array

    # 拆分数据集 & 分离data、label
    train_dataset, validation = train_test_split(train_data_array, test_size=0.2, random_state=21)
    train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.25, random_state=21) # 80% * 0.25 = 100% * 0.2

    train_dataset, train_label = utils.split_data_and_label(train_dataset)
    valid_dataset, valid_label = utils.split_data_and_label(validation)
    test_dataset, test_label = utils.split_data_and_label(test_dataset)

    return train_dataset, train_label, valid_dataset, valid_label, test_dataset, test_label, pred_dataset

def train_model(train_data, train_label, pred_data, params, en_amount):
    pred_all = 0
    for seed in range(en_amount):
        NFOLDS = 5
        kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
        kf = kfold.split(train_data, train_label)
        count = 0
        pred = np.zeros(pred_data.shape[0])
        valid_best_all = 0

        for i, (train_fold, validate) in enumerate(kf):
            print('Fold: ', i, ' training...')
            # K_Fold 分割数据集，验证集
            X_train, X_validate, label_train, label_validate = \
                train_data.iloc[train_fold, :], train_data.iloc[validate, :], train_label[train_fold], train_label[validate]
            # 数据集构造成lightgbm训练所需格式
            train = lgb.Dataset(X_train, label_train)
            valid = lgb.Dataset(X_validate, label_validate, reference=train)
            # 训练模型
            bst = lgb.train(params, train, num_boost_round=10000, valid_sets=valid, verbose_eval=-1,
                            early_stopping_rounds=50)
            pred += bst.predict(pred_data, num_iteration=bst.best_iteration)
            print(bst.best_score)
            # bst.best_score example : {'valid_0': {'rmse': 14.744103789371326}}
            valid_best_all += bst.best_score['valid_0']['rmse']
            count += 1
        pred /= NFOLDS
        valid_best_all /= NFOLDS
        print('cv score for valid is: ', 1 / (1 + valid_best_all))
        pred_all += pred

    pred_all /= en_amount
    return pred_all