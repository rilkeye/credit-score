#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke


import pandas as pd
import csv


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


def write_log(save_path, **arg):
    with open(save_path, 'a', encoding='utf-8') as f:
        for i in arg:
            f.write('\n' + i + ' : ' + str(arg[i]) + '\n')
        f.write('-' * 50 + '\n')

