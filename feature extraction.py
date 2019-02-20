#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import utils
import math


# 计算特征和类的平均值
def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean

# 计算Pearson系数
def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)	#计算x,y向量平均值
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

def calcSpearman(x,y):
    pass

# 计算每个特征的spearman系数，返回数组
def calcAttribute(dataSet):
    prr = []
    n = dataSet.shape[0] # 获取数据集行数和列数
    m = dataSet.shape[1]
    x = [0] * n             # 初始化特征x和类别y向量
    y = [0] * n
    for i in range(n):      # 得到类向量
        y[i] = dataSet[i][m-1]
    for j in range(m-1):    # 获取每个特征的向量，并计算Pearson系数，存入到列表中
        for k in range(n):
            x[k] = dataSet[k][j]
        prr.append(calcSpearman(x,y))
    return prr



train_dataset_path = 'data/train_dataset.csv'
train_data_array = utils.build_data_array(train_dataset_path, tag='train')
score = utils.get_array_column(train_data_array, -1)
for i in range(train_data_array.shape[1] - 1):
    feature = utils.get_array_column(train_data_array, i)
    pearson = calcPearson(feature, score)
    print('Feature{}\'s Pearson is:{}'.format(i, pearson))


def get_Pearson():
    pass

