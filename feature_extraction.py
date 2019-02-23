#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import utils

import pandas as pd
import numpy as np
import math
import csv
from sklearn import metrics as mr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
import pylab as mpl


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
    score = utils.get_array_column(array, -1)
    # 获取特征标签
    with open('data/train_dataset.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
            if idx == 0:
                firstline = line  # 首行，特征标签行

    del firstline[-1] # 删除“信用分”
    del firstline[0] # 删除“用户编码”

    # 计算特征Pearson相关系数
    for i in range(array.shape[1] - 1):
        feature = utils.get_array_column(array, i)
        pearson_i = cal_Pearson(feature, score)
        Pearson[str(firstline[i])] = pearson_i
        if i == 14:
            print(cal_Pearson(feature, utils.get_array_column(array, i+1)))
    A = np.zeros(50000,)
    for No in range(21,28):
        fea = utils.get_array_column(array, i)
        A += fea

    print(cal_Pearson(A, score))

    return Pearson

        # with open('data/pearson.txt', 'a', encoding='utf-8') as f:
        #    f.write('Pearson correlation of feature {} "{}" is : {}'.format(i, firstline[i], pearson_i))
        #    f.write('\n')

def get_MI(array):
    '''
    计算信息增益MI
    :param array: 最后一列为label或score的矩阵
    '''
    score = utils.get_array_column(array, -1)
    # 获取特征标签
    with open('data/train_dataset.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
            if idx == 0:
                firstline = line  # 首行，特征标签行

    del firstline[-1]  # 删除“信用分”
    del firstline[0]  # 删除“用户编码”

    for i in range(array.shape[1] - 1):
        feature = utils.get_array_column(array, i)
        MI = mr.mutual_info_score(feature, score)
        print('MI of feature {} "{}" is : {}'.format(i, firstline[i], MI))
    return MI

def LassoCV():
    train = pd.read_csv('data/train_dataset.csv', header=0)  # Load the train file into a dataframe
    df = pd.get_dummies(train.iloc[:, 1:-1])  # 去除第一列“用户编码”和最后一列“信用分”
    df = df.fillna(df.mean())

    X_train = df
    y = train.信用分

    # 调用LassoCV函数，并进行交叉验证，默认cv=3
    model_lasso = LassoCV(alphas=[0.1, 1, 0.001, 0.0005]).fit(X_train, y)

    # 模型所选择的最优正则化参数alpha
    print(model_lasso.alpha_)

    # 各特征列的参数值或者说权重参数，为0代表该特征被模型剔除了
    print(model_lasso.coef_)

    # 输出看模型最终选择了几个特征向量，剔除了几个特征向量
    coef = pd.Series(model_lasso.coef_, index=X_train.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
        sum(coef == 0)) + " variables")

    # 输出所选择的最优正则化参数情况下的残差平均值，因为是3折，所以看平均值
    print(np.sqrt(-cross_val_score(model_lasso, X_train, y, scoring="neg_mean_squared_error", cv = 3)).mean())

    # 画出特征变量的重要程度，这里面选出前17个重要，后10个不重要的举例
    imp_coef = pd.concat([coef.sort_values().head(17),
                          coef.sort_values().tail(10)])

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图片中文为'-'，显示图片中文为方框的问题
    matplotlib.rcParams['figure.figsize'] = (10.0, 30.0)
    imp_coef.plot(kind="barh")
    plt.title("Coefficients in the Lasso Model")
    plt.show()


# LassoCV()

train_dataset_path = 'data/train_dataset.csv'
train_data_df = utils.load_data(train_dataset_path, tag='train')
train_data_array = train_data_df.values
score = utils.get_array_column(train_data_array, -1)

feature = train_data_df[['当月通话交往圈人数']].values
label = train_data_df[['信用分']].values
# print(feature, label)

plt.scatter(feature, label)
plt.show()

# 计算各项特征的Pearson相关系数，返回 Dict
# pearson = utils.get_Pearson(train_data_array)

# 计算各项特征的互信息MI，
# MI = utils.get_MI(train_data_array)



