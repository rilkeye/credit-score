#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Rilke

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def _lgb_tree(train_data, train_label, valid_data, valid_label, pred_data, params):
    train = lgb.Dataset(train_data, train_label)
    valid = lgb.Dataset(valid_data, valid_label, reference=train)

    gbm = lgb.train(params, train, num_boost_round=10000, verbose_eval=-1, early_stopping_rounds=50)

    valid_pred = gbm.predict(valid_data)
    pred = gbm.predict(pred_data)
    valid_mae = mean_absolute_error(valid_pred, valid_data)
    print('')