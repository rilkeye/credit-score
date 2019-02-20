#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import utils

train_dataset_path = 'data/train_dataset.csv'
train_data_array = utils.build_data_array(train_dataset_path, tag='train')

# 计算各项特征的Pearson相关系数，返回 Dict
pearson = utils.get_Pearson(train_data_array)





