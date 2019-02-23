#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import numpy as np
import pandas as pd
import seaborn as sns
import utils
import matplotlib as plt

# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

def visualize_df(df):
    print(df.head())
    print(df.describe())
    print(sns.countplot(df['是否4G不健康客户']))
    missing_data = draw_missing_data_table(df)
    print(missing_data)

path = 'data/train_dataset.csv'
df = utils.load_data(path, 'train')
visualize_df(df)

