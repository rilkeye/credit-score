#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import utils

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
import pylab as mpl


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 3))
    return(rmse)

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
    print(rmse_cv(model_lasso).mean())

    # 画出特征变量的重要程度，这里面选出前17个重要，后10个不重要的举例
    imp_coef = pd.concat([coef.sort_values().head(17),
                          coef.sort_values().tail(10)])

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图片中文为'-'，显示图片中文为方框的问题
    matplotlib.rcParams['figure.figsize'] = (10.0, 30.0)
    imp_coef.plot(kind="barh")
    plt.title("Coefficients in the Lasso Model")
    plt.show()


if __name__ == '__main':
    LassoCV()


    train_dataset_path = 'data/train_dataset.csv'
    train_data_array = utils.build_data_array(train_dataset_path, tag='train')
    score = utils.get_array_column(train_data_array, -1)

    # 计算各项特征的Pearson相关系数，返回 Dict
    pearson = utils.get_Pearson(train_data_array)

    # 计算各项特征的互信息MI，
    MI = utils.get_MI(train_data_array)

