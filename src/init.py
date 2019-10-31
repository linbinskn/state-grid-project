#############################################
#        load data from database            #
#           暂时直接读本地文件               #
#############################################
import pandas as pd
import os
from os.path import join
from utils import *


def load_data(filepath, filename):
    print('Read data from {}'.format(filename))
    read_dir = join(filepath, filename)
    try:
        read_data = pd.read_csv(read_dir)
    except:
        read_data = pd.read_csv(read_dir, encoding = 'gb18030')
    return read_data


@timeit
def load_fulldata():
    full_data = {}
    rootdir = join(os.getcwd(), 'dataset')
    rootdir = 'C:\\tsinghua\\大数据系统基础B\\state grid project\\dataset'
    datanames = os.listdir(rootdir)

    for dataname in datanames:
        full_data[dataname.split('.')[0]] = load_data(rootdir, dataname)
    return full_data
