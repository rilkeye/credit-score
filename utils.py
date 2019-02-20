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
    dataframe.to_csv('data/submit2.csv',index=False, sep=',', encoding='utf-8')
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

# 计算特征和类的平均值
def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean

# 计算Pearson系数
def cal_Pearson(x,y):
    x_mean,y_mean = calcMean(x,y) # 计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p

def get_Pearson(array):
    '''
    :return: Pearson dict
    '''
    Pearson = {}
    score = get_array_column(array, -1)
    # 获取特征标签
    with open('data/train_dataset.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
            if idx == 0:
                firstline = line  # 首行，特征标签行

    del firstline[-1]
    del firstline[0]

    # 计算特征Pearson相关系数
    for i in range(array.shape[1] - 1):
        feature = get_array_column(array, i)
        pearson_i = cal_Pearson(feature, score)
        Pearson[str(firstline[i])] = pearson_i

    return Pearson

        # with open('data/pearson.txt', 'a', encoding='utf-8') as f:
        #    f.write('Pearson correlation of feature {} "{}" is : {}'.format(i, firstline[i], pearson_i))
        #    f.write('\n')



