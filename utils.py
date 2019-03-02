#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import numpy as np
import pandas as pd
import csv
import time
from sklearn.preprocessing import OneHotEncoder


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


def one_hot(df, feature):
    tmpdata = df[[feature]]
    df = df.drop(columns=[feature])
    enc = OneHotEncoder()
    enc.fit(tmpdata)
    tmpdata = enc.transform(tmpdata).toarray()
    columns_list = []
    for i in range(int(enc.n_values_)):
        col_name = feature + str(i)
        columns_list.append(col_name)
    tmpdata = pd.DataFrame(tmpdata, columns=columns_list)
    df = pd.concat([df, tmpdata], axis=1)
    return df

def timer():
    return time.time()

def processed_df(df):
    '''
    Processing, building, deleting feature in the function...
    :param df:  dataframe
    :return:  dataframe
    '''

    # build new feature
    # df['高端消费评分加权'] = 2 * df['当月是否景点游览'] + df['是否经常逛商场的人'] + df['当月是否逛过福州仓山万达'] + \
    #                       3 * df['当月是否到过福州山姆会员店'] + df['当月是否看电影'] + 2 * df['当月是否体育场馆消费']
    df['是否去过商场'] = df['当月是否逛过福州仓山万达'] + df['当月是否到过福州山姆会员店']
    df['是否去过商场'] = df['是否去过商场'].map(lambda x: 1 if x >= 1 else 0)
    df['是否_商场电影'] = df['是否去过商场'] * df['当月是否看电影']
    df['是否_商场旅游'] = df['是否去过商场'] * df['当月是否景点游览']
    df['是否_商场体育馆'] = df['是否去过商场'] * df['当月是否体育场馆消费']
    df['是否_电影体育馆'] = df['当月是否看电影'] * df['当月是否体育场馆消费']
    df['是否_旅游体育馆'] = df['当月是否景点游览'] * df['当月是否体育场馆消费']
    df['是否_商场旅游体育馆'] = df['是否去过商场'] * df['当月是否景点游览'] * df['当月是否体育场馆消费']
    df['是否_商场电影体育馆'] = df['是否去过商场'] * df['当月是否看电影'] * df['当月是否体育场馆消费']
    df['是否_商场电影旅游'] = df['是否去过商场'] * df['当月是否看电影'] * df['当月是否景点游览']
    df['是否_体育馆电影旅游'] = df['当月是否体育场馆消费'] * df['当月是否看电影'] * df['当月是否景点游览']
    df['是否_商场体育馆电影旅游'] = df['是否去过商场'] * df['当月是否体育场馆消费'] * df['当月是否看电影'] * df['当月是否景点游览']

    df['话费稳定性'] = df['用户账单当月总费用（元）'] / (df['用户近6个月平均消费值（元）'] + 5)

    # df['近三个月月均商场出现次数(归一化)'] = normalize(df['近三个月月均商场出现次数'])
    # df = df.drop(columns=['近三个月月均商场出现次数'])

    df['用户历史消费总额'] = df['用户网龄（月）'] * df['用户近6个月平均消费值（元）']

    df['用户消费趋势'] = np.where((df['用户账单当月总费用（元）'] - df['用户近6个月平均消费值（元）']) >= 0, 1, 0)

    df['用户不良记录种类'] = df['缴费用户当前是否欠费缴费'] + df['是否黑名单客户']
    # df['用户是否有不良消费记录'] = np.where((df['缴费用户当前是否欠费缴费']  | df['是否黑名单客户']), 1, 0)

    df['飞机'] = 0
    df['飞机'][df['当月飞机类应用使用次数'] != 0] = 1
    # df = df.drop(columns=['当月飞机类应用使用次数'])

    df['火车'] = 0
    df['火车'][df['当月火车类应用使用次数'] != 0] = 1
    # df = df.drop(columns=['当月火车类应用使用次数'])

    # df['旅游'] = 0
    # df['旅游'][(df['当月旅游资讯类应用使用次数'] > 0) & (df['当月旅游资讯类应用使用次数'] <= 50)] = 1
    # df['旅游'][(df['当月旅游资讯类应用使用次数'] > 50)] = 2

    df['出行指数'] = df['火车'] + 2 * df['飞机']
    df = df.drop(columns=['火车', '飞机'])

    # 用户充值是否整数
    def is_int(n):
        n = round(n)
        if n % 100 == 0:
            return 0
        if n % 10 == 0:
            return 1
        return 2

    df['用户充值是否整数'] = df['缴费用户最近一次缴费金额（元）'].map(is_int)

    df.loc[df['用户年龄'] == 0, '用户年龄'] = df['用户年龄'].mode()

    transform_value_feature = ['用户年龄', '用户网龄（月）', '当月通话交往圈人数', '近三个月月均商场出现次数',
                               '当月网购类应用使用次数', '当月物流快递类应用使用次数', '当月金融理财类应用使用总次数',
                               '当月视频播放类应用使用次数', '当月飞机类应用使用次数', '当月火车类应用使用次数',
                               '当月旅游资讯类应用使用次数']
    log_features = ['当月网购类应用使用次数', '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数']
    user_bill_features = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）', '用户账单当月总费用（元）',
                         '用户当月账户余额（元）']
    for col in user_bill_features + log_features:
        df[col] = df[col].map(lambda x: np.log1p(x))

    for col in transform_value_feature:
        max_limit = np.percentile(df[col].values, 99.9)
        min_limit = np.percentile(df[col].values, 0.1)
        df[col].loc[df[col] > max_limit] = max_limit
        df[col].loc[df[col] < min_limit] = min_limit

    #one-hot encoding
    # df = one_hot(df, '高端消费评分加权')
    df = one_hot(df, '用户话费敏感度')
    df = one_hot(df, '用户不良记录种类')
    df = one_hot(df, '用户充值是否整数')
    # df = one_hot(df, '用户年龄')


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

def data_cleaning(data):
    # copy from github : https://github.com/HoGiggle/consumer_credict
    user_int_fea = ['用户年龄', '用户网龄（月）', '当月通话交往圈人数', '近三个月月均商场出现次数', '当月网购类应用使用次数',
                    '当月物流快递类应用使用次数', '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数', '当月飞机类应用使用次数',
                    '当月火车类应用使用次数', '当月旅游资讯类应用使用次数']

    call_cost_fea = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）', '用户账单当月总费用（元）', '用户当月账户余额（元）']

    user_big_int_fea = ['当月网购类应用使用次数', '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数']

    # 异常值截断
    for col in user_int_fea + call_cost_fea:
        high = np.percentile(data[col].values, 99.8)
        low = np.percentile(data[col].values, 0.2)
        data.loc[data[col] > high, col] = high
        data.loc[data[col] < low, col] = low

    #     # 过大量级值取log平滑
    for col in user_big_int_fea:
        data[col] = data[col].map(lambda x: np.log1p(x))

    # 交通APP特征汇总
    data['交通APP次数'] = data['当月火车类应用使用次数'] + data['当月飞机类应用使用次数']
    #     data = data.drop(columns=['当月火车类应用使用次数', '当月飞机类应用使用次数'])

    # 高档消费
    data['高档消费'] = data['当月是否逛过福州仓山万达'] + data['当月是否到过福州山姆会员店']
    #     data = data.drop(columns=['当月是否逛过福州仓山万达', '当月是否到过福州山姆会员店'])

    # 缴费金额
    data['充值方式'] = 0
    data['充值方式'][(data['缴费用户最近一次缴费金额（元）'] % 10 == 0) &
                 data['缴费用户最近一次缴费金额（元）'] != 0] = 1

    data['缴费习惯'] = data['缴费用户最近一次缴费金额（元）'] / (data['用户近6个月平均消费值（元）'] + 1)

    # 当月话费/6月话费, 最近消费稳定性
    data['最近账单稳定性'] = data['用户账单当月总费用（元）'] / (data['用户近6个月平均消费值（元）'] + 1)

    # 当月话费/当月账户余额
    data['账户余额利用率'] = data['用户账单当月总费用（元）'] / (data['用户当月账户余额（元）'] + 1)

    # 当月欠费
    #     data['当月欠费'] = 0
    #     data['当月欠费'][(data['用户最近一次缴费距今时长（月）'] == 0) & (data['缴费用户当前是否欠费缴费'] == 1)] = 1

    # 根据年龄区分社会角色，不确定，<18，[18,  23），[23,  30)，[30，45），[45, +)
    data['不确定角色'] = 0
    data['不确定角色'][data['用户年龄'] == 0] = 1

    data['小学生'] = 0
    data['小学生'][(data['用户年龄'] > 0) & (data['用户年龄'] < 18) & (data['是否大学生客户'] == 0)] = 1

    data['大学生'] = 0
    data['大学生'][((data['用户年龄'] >= 18) & (data['用户年龄'] < 23)) | (data['是否大学生客户'] == 1)] = 1

    data['工作10年内'] = 0
    data['工作10年内'][((data['用户年龄'] >= 23) & (data['用户年龄'] < 30))] = 1

    data['工作10年上'] = 0
    data['工作10年上'][((data['用户年龄'] >= 30) & (data['用户年龄'] < 45))] = 1

    data['老干部'] = 0
    data['老干部'][(data['用户年龄'] >= 45)] = 1
    #     data['社会角色'] = -1
    #     data['社会角色'][data['用户年龄'] == 0]
    #     data['社会角色'][((data['用户年龄'] > 0) & (data['用户年龄'] < 18)) & (data['是否大学生客户'] == 0)] = 1
    #     data['社会角色'][((data['用户年龄'] >= 18) & (data['用户年龄'] < 23)) | (data['是否大学生客户'] == 1)] = 2
    #     data['社会角色'][((data['用户年龄'] >= 23) & (data['用户年龄'] < 30))] = 3
    #     data['社会角色'][((data['用户年龄'] >= 30) & (data['用户年龄'] < 45))] = 4
    #     data['社会角色'][((data['用户年龄'] >= 45))] = 5
    data = data.drop(columns=['是否大学生客户'], axis=1)

    # 是否新互联网用户 用户网龄分段，<12，[12,  36)，[36,  120)，[120,  +)
    data['新用户'] = 0
    data['新用户'][(data['用户网龄（月）'] < 12)] = 1

    data['3年用户'] = 0
    data['3年用户'][(data['用户网龄（月）'] >= 12) & (data['用户网龄（月）'] < 36)] = 1

    data['10年用户'] = 0
    data['10年用户'][(data['用户网龄（月）'] >= 36) & (data['用户网龄（月）'] < 120)] = 1

    data['老用户'] = 0
    data['老用户'][(data['用户网龄（月）'] >= 120)] = 1

    #     data['用户入网深度'] = 0
    #     data['用户入网深度'][(data['用户网龄（月）'] >= 12) & (data['用户网龄（月）'] < 36)] = 1
    #     data['用户入网深度'][(data['用户网龄（月）'] >= 36) & (data['用户网龄（月）'] < 120)] = 2
    #     data['用户入网深度'][data['用户网龄（月）'] >= 120] = 3
    #     data = data.drop(columns=['用户网龄（月）'])

    return data