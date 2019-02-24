#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import numpy as np
import pandas as pd
import csv


def write_log(save_path, **arg):
    with open(save_path, 'a', encoding='utf-8') as f:
        for i in arg:
            f.write('\n' + i + ' : ' + str(arg[i]) + '\n')
        f.write('-' * 50 + '\n')


def normalize(df_col):
    '''
    将某一列数据归一化
    :param df_col:  a column of dataframe
    :return:  a column of dataframe
    '''
    df = (df_col - df_col.min()) / (df_col.max() - df_col.min())
    return df


def processed_df(df):
    '''
    Processing, building, deleting feature in the function...
    :param df:  dataframe
    :return:  dataframe
    '''

    # build new feature
    df['高端消费评分'] = 2 * df['当月是否景点游览'] + 2 * df['是否经常逛商场的人'] + df['当月是否逛过福州仓山万达'] + \
                   3 * df['当月是否到过福州山姆会员店'] + df['当月是否看电影'] + 3 * df['当月是否体育场馆消费']

    df['话费稳定性'] = df['用户账单当月总费用（元）'] / (df['用户近6个月平均消费值（元）'] + 5)

    df['近三个月月均商场出现次数(归一化)'] = normalize(df['近三个月月均商场出现次数'])

    df['用户历史消费总额'] = df['用户网龄（月）'] * df['用户近6个月平均消费值（元）']

    df['用户消费趋势'] = np.where((df['用户账单当月总费用（元）'] - df['用户近6个月平均消费值（元）']) >= 0, 1, 0)

    df['用户不良记录种类'] = df['缴费用户当前是否欠费缴费'] + df['是否4G不健康客户'] + df['是否黑名单客户']

    df['飞机'] = 0
    df['飞机'][df['当月飞机类应用使用次数'] != 0] = 1
    # df = df.drop(columns=['当月飞机类应用使用次数'])

    df['火车'] = 0
    df['火车'][df['当月火车类应用使用次数'] != 0] = 1
    # df = df.drop(columns=['当月火车类应用使用次数'])

    df['旅游'] = 0
    df['旅游'][(df['当月旅游资讯类应用使用次数'] > 0) & (df['当月旅游资讯类应用使用次数'] <= 50)] = 1
    df['旅游'][(df['当月旅游资讯类应用使用次数'] > 50)] = 2
    # df = df.drop(columns=['当月旅游资讯类应用使用次数'])

    df['出行指数'] = df['火车'] + df['飞机'] + df['旅游']
    df = df.drop(columns=['火车', '飞机', '旅游'])

    df['用户账单当月总费用（元）'] = np.log1p(df['用户账单当月总费用（元）'])
    # df['用户当月账户余额（元）'] = np.log1p(df['用户当月账户余额（元）'])
    # df['当月通话交往圈人数'] = np.log1p(df['当月通话交往圈人数'])

    # a = [0, 18, 60]
    # df['年龄阶层'] = pd.cut(df['用户年龄'], bins=a, label=[1,2,3])
    # df = df.drop(columns=['用户年龄'])
    # df['用户年龄'][df['用户年龄'] == 0] = df['用户年龄'].mean()
    # df['用户年龄'][df['用户年龄'] <= 18] = None
    # df = df.fillna(df['用户年龄'].mean())

    # Drop useless columns
    df = df.drop(columns=['用户实名制是否通过核实'])
    df = df.drop(columns=['当月是否景点游览', '是否经常逛商场的人', '当月是否逛过福州仓山万达', '当月是否到过福州山姆会员店'
                          , '当月是否看电影', '当月是否体育场馆消费'])
    df = df.drop(columns=['近三个月月均商场出现次数'])

    # tried but useless

    '''
    df['是否线下充值'] = 0
    df['是否线下充值'][(df['缴费用户最近一次缴费金额（元）'] % 10 == 0) & (df['缴费用户最近一次缴费金额（元）'] != 0)] = 1

    df['用户年龄'][df['用户年龄'] >= 90] = None
    df['用户年龄'][df['用户年龄'] <= 18] = None
    df = processed_df.fillna(df['用户年龄'].mean())

    df['用户互联网活跃度'] = df['当月网购类应用使用次数'] + df['当月物流快递类应用使用次数'] + \
                                     df['当月金融理财类应用使用总次数'] + df['当月视频播放类应用使用次数'] + \
                                     df['当月飞机类应用使用次数'] + df['当月火车类应用使用次数'] + \
                                     df['当月旅游资讯类应用使用次数']

    df['用户互联网活跃度'] = np.log(df['用户互联网活跃度'])
    print(df['用户互联网活跃度'])
    df = df.drop(columns=['当月网购类应用使用次数', '当月物流快递类应用使用次数', '当月金融理财类应用使用总次数',
                                              '当月视频播放类应用使用次数', '当月飞机类应用使用次数', '当月火车类应用使用次数',
                                              '当月旅游资讯类应用使用次数'])


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
