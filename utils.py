#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import numpy as np
import pandas as pd
import time
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

def write_log(num_round, params, model_path, score):
    with open('training log.txt', 'a', encoding='utf-8') as f:
        f.write('Time : ' + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + '\n')
        f.write('Num_round : {}'.format(num_round) + '\n' + '\n')
        f.write('Params : {}'.format(params) + '\n' + '\n')
        f.write('Score : {}'.format(score) + '\n')
        f.write('Model saved path : {}'.format(model_path) + '\n')
        f.write('-----------------------------------------------------' + '\n')

