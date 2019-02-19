#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import numpy as np
import pandas as pd
import csv

def build_data_array(path, tag):
    '''

    :param path: file path
    :param tag: "train" or "test"
    :return: test array : (50000, 28) ; train array : (50000, 29)
    '''
    train_data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            train_data.append(line)

    del (train_data[0])  # 删除.csv文件首行id信息

    if tag == 'test':
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