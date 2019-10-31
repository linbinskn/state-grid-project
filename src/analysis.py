#############################################
#              data analysis                #
#############################################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import *

"""
    Usage:: analyse electricity in a single day
    :param data: data from file1csv
    :param user_id: id from column USERID
    :param day: day from column DAY
    :return :None
    :rtype: None
"""
def perdayanalysis(data, user_id, day):
    userid_data = data[data['USERID'] == user_id]
    day_data = userid_data[userid_data['DAY'] == day]
    x = list(np.linspace(1, 24, 48))
    y = np.array(day_data)[0, 2:]

    plt.plot(x, y)
    plt.show()

"""
    Usage:: analyse electricity in one and a half year
    :param data: data from file1csv
    :param user_id: id from column USERID
    :return :None
    :rtype: None
"""
def peryearanalysis(data, user_id):
    user_id_data = data[data['USERID'] == user_id]
    user_id_data.sort_values(axis=0, by='DAY', inplace=True)
    x = list(user_id_data['DAY'])
    user_id_data = user_id_data.iloc[:,2:]
    y = list(user_id_data.sum(axis=1))

    plt.plot(x, y)
    plt.show()
