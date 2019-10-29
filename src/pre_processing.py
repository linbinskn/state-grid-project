#############################################
#   data cleaning and feature engineering   #
#############################################

import pandas as pd
import numpy as np
from utils import *


"""
    Usage:: fill NAN with mean
    :param data: data from file1csv
    :return :data
    :rtype: object(pandas)
"""
def fill_nan_mean(data):
    values = dict([(colname, colvalue) for colname, colvalue in zip(data.columns.tolist(), data.mean().tolist())])
    data.fillna(value=values, inplace=True)
    return data


"""
    Usage:: fill NAN with preceding value
    :param data: data from file1csv
    :return :data
    :rtype: object(pandas)
"""
def fill_nan_ffill(data):
    data.fillna(method='ffill', inplace=True)
    return data


"""
    Usage:: fill NAN with single_user_mean
    :param data: data from file1csv
    :return :data
    :rtype: object(pandas)
"""
def fill_nan_user(data):
    data = data.groupby('USERID').apply(lambda x:fill_userid_group(x))
    return data


def fill_userid_group(group):
    values = dict([(colname, colvalue) for colname, colvalue in zip(group.columns.tolist(),group.mean().tolist())])
    group.fillna(value=values, inplace=True)
    return group

"""
    Usage:: split data into train and test
    :param data: full data including file1csv and four questionnaires
    :return :train_data, test_data
    :rtype: object(pandas)
"""
@timeit
def traindata_split(full_data):
    split_data = full_data['file1csv']
    train_data = pd.DataFrame(columns=full_data.columns)
    test_data = pd.DataFrame(columns=full_data.columns)
    for day in range(565, 731, 1):
        day_data = split_data[split_data['DAY'] == day]
        test_data = pd.concat([test_data, day_data])
    for day in range(195, 565, 1):
        day_data = split_data[split_data['DAY'] == day]
        train_data = pd.concat([train_data, day_data])
    return train_data, test_data


