#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import numpy as np
import pandas as pd
import csv
import math

def load_data(path, tag):
    '''

    :return: <class 'pandas.core.frame.DataFrame'>
             [50000 rows x 28 columns] when tag == 'pred' , test文件不含“信用分”列
             [50000 rows x 29 columns]  when tag == 'train' , train文件含“信用分”列
    '''
    data = pd.read_csv(path, header=0)
    if tag == 'train':
        dataframe = pd.get_dummies(data.iloc[:, 1:]) # 去除第一列“用户编码”数据
        return dataframe

    if tag == 'pred':
        dataframe = pd.get_dummies(data.iloc[:, 1:]) # 去除第一列“用户编码”数据
        return dataframe

    else :
        raise()

def split_data_and_label(array):
    '''
    输入的array需要确保label在最后一列
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

