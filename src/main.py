import pandas as pd
import init
import os
from os.path import join
from utils import *
import model
import pre_processing
"""
    Usage:: load file1csv and four questionnaires
    :param None
    :return :5 tables in form of pandas
    :rtype: list
"""


def main():
    full_data = init.load_fulldata()
    timeseries_model = model.Model()
    train_data, test_data = pre_processing.traindata_split(full_data)
    timeseries_model.fit(train_data)
    results = timeseries_model.predict(test_data)
if __name__ == '__main__':
    main()

full_data = pd.read_csv("C:/tsinghua/大数据系统基础B/grin project/dataset/file1csv.csv")