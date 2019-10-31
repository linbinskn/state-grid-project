# 国家电网大数据平台(初级版本)
### how to start
`python main.py`
## 文件介绍
#### init.py
- load_data:读取单个数据集
- load_fulldata:读取全部数据集

#### analysis.py
- perdayanalysis:分析某一天的用电量趋势，并画出图像，需要输入具体的用户id和哪一天
- peryearanalysis:分析某一年的用电趋势，并画出图像，需要输入具体的用户id

#### model.py
- fit:特征工程加训练
- predict：预测结果

#### pre_processing.py
- fill_nan_mean:使用所有用户当天的平均值填补数据
- fill_nan_ffill:使用前项数据填补缺失值
- traindata_split:将1.5年分为1年数据集和0.5年测试集

#### utils.py
- timeit:函数计时器，用于时间长度计算