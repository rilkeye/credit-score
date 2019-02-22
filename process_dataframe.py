#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import numpy as np
import pandas as pd
import csv
import math


def processed_df(df):
    '''
    Processing, building, deleting feature in the function...
    :param df:  dataframe
    :return:  dataframe
    '''
    df['进行过多少种高端消费（max：7）'] = df['当月是否景点游览'] + df['是否经常逛商场的人'] + df['当月是否逛过福州仓山万达'] + \
                              df['当月是否到过福州山姆会员店'] + df['当月是否看电影'] + df['当月是否景点游览'] + \
                              df['当月是否体育场馆消费']

    df = df.drop(columns=['当月是否景点游览', '是否经常逛商场的人', '当月是否逛过福州仓山万达', '当月是否到过福州山姆会员店',
                          '当月是否看电影', '当月是否景点游览', '当月是否体育场馆消费', '用户实名制是否通过核实'])

    df['用户历史消费总额'] = df['用户网龄（月）'] * df['用户近6个月平均消费值（元）']
    df['用户消费趋势'] = np.where((df['用户账单当月总费用（元）'] - df['用户近6个月平均消费值（元）']) >= 0, 1, 0)
    df['用户不良记录种类'] = df['缴费用户当前是否欠费缴费'] + df['是否4G不健康客户'] + df['是否黑名单客户']

    # Drop useless columns
    df = df.drop(columns=['用户实名制是否通过核实'])

    '''
       processed_df['用户年龄'][processed_df['用户年龄'] >= 90] = None
       processed_df['用户年龄'][processed_df['用户年龄'] <= 18] = None
       processed_df = processed_df.fillna(processed_df['用户年龄'].mean())
       '''

    '''
    processed_df['用户互联网活跃度'] = processed_df['当月网购类应用使用次数'] + processed_df['当月物流快递类应用使用次数'] + \
                                     processed_df['当月金融理财类应用使用总次数'] + processed_df['当月视频播放类应用使用次数'] + \
                                     processed_df['当月飞机类应用使用次数'] + processed_df['当月火车类应用使用次数'] + \
                                     processed_df['当月旅游资讯类应用使用次数']

    processed_df['用户互联网活跃度'] = np.log(processed_df['用户互联网活跃度'])
    print(processed_df['用户互联网活跃度'])
    processed_df = processed_df.drop(columns=['当月网购类应用使用次数', '当月物流快递类应用使用次数', '当月金融理财类应用使用总次数',
                                             '当月视频播放类应用使用次数', '当月飞机类应用使用次数', '当月火车类应用使用次数',
                                             '当月旅游资讯类应用使用次数'])
    '''

    '''
    def func1(age):
        if age > 25 and age < 55:
            return 3
        elif age < 18 or age > 65:
            return 1
        else:
            return 2

    processed_df['用户消费能力等级（三级最高）'] = processed_df['用户年龄'].map(func1)
    processed_df = processed_df.drop(columns=['用户年龄'])
    '''

    return df