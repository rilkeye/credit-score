#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke


import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def train(train_data, train_label, pred_data, params):
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=21)
    train_label, valid_lebal = train_test_split(train_label, test_size=0.2, random_state=21)

    print( 'training...')
    # 数据集构造成lightgbm训练所需格式
    train = lgb.Dataset(train_data, train_label)
    valid = lgb.Dataset(valid_data, valid_lebal, reference=train)
    # 训练模型
    bst = lgb.train(params, train, num_boost_round=10000, valid_sets=valid, verbose_eval=-1,
                    early_stopping_rounds=50)
    # 预测
    pred = bst.predict(pred_data, num_iteration=bst.best_iteration)
    # bst.best_score example : {'valid_0': {'l1': 14.744103789371326}}
    valid_score = bst.best_score['valid_0']['l1']

    return pred, valid_score




def make_predtion(dataset, model_path):
    '''
    :param dataset: 需要预测的数据
    :param model_path: 模型路径
    :return:
    '''

    bst = lgb.Booster(model_file=model_path)
    pred = bst.predict(dataset)

    return pred


def handle_data(train_dataset):
    # 拆分为训练集、验证集、测试集 & 分离data、label
    train_data, valid_data = train_test_split(train_dataset, test_size=0.2, random_state=21)
    train_data, test_data = train_test_split(train_data, test_size=0.25, random_state=21) # 80% * 0.25 = 100% * 0.2

    train_label = train_data['信用分']
    valid_label = valid_data['信用分']
    test_label = test_data['信用分']
    train_data = train_data.drop(columns=['信用分'])
    valid_data = valid_data.drop(columns=['信用分'])
    test_data = test_data.drop(columns=['信用分'])
    return train_data, train_label, valid_data, valid_label, test_data, test_label


def train2(train_data, train_label, pred_data, params, en_amount):
    pred_all = 0
    valid_score = 0
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
            # bst.best_score example : {'valid_0': {'l1': 14.744103789371326}}
            valid_best_all += bst.best_score['valid_0']['l1']
            count += 1
        pred /= NFOLDS
        valid_best_all /= NFOLDS
        print('cv score for valid is: ', 1 / (1 + valid_best_all))
        pred_all += pred
        valid_score += valid_best_all

    pred_all /= en_amount
    valid_score /= en_amount
    return pred_all, valid_score

