#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import numpy as np
import pandas as pd
import csv
import math


def build_data_array(path, tag):
    '''
    训练模型时，通过调用该函数构造特征数据（与信用分)矩阵。Ps：载入所有特征
    :param path: file path
    :param tag: "train" or "test"
    :return: test array : (50000, 28) ; train array : (50000, 29)
    '''
    train_data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            train_data.append(line)

    del (train_data[0])  # 删除.csv文件首行标签信息

    if tag == 'pred':
        array = np.zeros(shape=(50000,28))
        for row_idx in range(50000):
            for col_idx in range(1, 29):
                array[row_idx][col_idx - 1] = float(train_data[row_idx][col_idx])
        return array

    if tag == 'train':
        array = np.zeros(shape=(50000,29))
        for row_idx in range(50000):
            for col_idx in range(1, 30):
                array[row_idx][col_idx - 1] = float(train_data[row_idx][col_idx])
        return array
    else:
        raise()

def split_data_and_label(array):
    '''

    :param array: shape = (m, n)
    :return: array.shape = (m, n-1) ; label.shape = (m,)
    '''
    label = array[:,-1] # 取array最后一列数据
    array = np.delete(array, -1, axis=1) # del last column of array
    return array, label

def write_SubmitionFile(result, id_path):
    score = result
    with open(id_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        id = [row[0] for row in reader]
    del id[0]

    dataframe = pd.DataFrame({'id':id, 'score':score})
    dataframe.to_csv('data/submit.csv',index=False, sep=',', encoding='utf-8')
    print('Submitfile had saved...')

def give_a_mark(pred, label):
    '''
    MAE = 1/n * Sum(abs(pred(i) - y(i)))
    Score = 1 / (1 + MAE)
    '''
    pred = np.array(pred)
    label = np.array(label)
    MAE = sum(abs(pred-label)) / float(len(pred))
    score = 1.0 / (1.0 + MAE)
    return score

def get_array_column(array, idx):
    '''
    :param idx: the column index we need...
    :return:  list of some column.
    '''
    list = array[:,idx]
    return list

def write_log(save_path, **arg):
    with open(save_path, 'a', encoding='utf-8') as f:
        for i in arg:
            f.write('\n' + i + ' : ' + str(arg[i]) + '\n')
        f.write('-' * 50 + '\n')

