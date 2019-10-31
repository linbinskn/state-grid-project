#############################################
#   model training and predicting           #
#############################################
import numpy as np
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dropout
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class Model:
    n_steps = 1       #LSTM param
    n_features = 1    # LSTM para
    # m
    def split_sequence(self, sequence, n_steps):
        #将数据按窗口分为训练集测试集
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence) - 1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)


    def mape(self, y_true, y_pred):
        """
        参数:
        y_true -- 测试集目标真实值
        y_pred -- 测试集目标预测值
        返回:
        mape -- MAPE 评价指标
        """
        n = len(y_true)
        mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
        return mape


    def create_lstmmodel(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(512, activation='relu'), input_shape=(self.n_steps, self.n_features)))
        model.add(Dense(1))
        # model.add(Dropout(0.1))
        model.compile(optimizer='adam', loss='mse',metrics=['mse'])
        return model

    @timeit
    def LSTM_train(self, train_data):
        data = train_data[train_data['USERID'] == 1000].drop('USERID', axis=1).set_index(keys='DAY').sum(axis=1)
        data = data.sort_index()
        data = data.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data.reshape(-1, 1))
        train_data = data[:450]
        test_data = data[450:]
        #
        X_train, y_train = self.split_sequence(train_data, self.n_steps)
        X_test, y_test = self.split_sequence(test_data, self.n_steps)
        n_features = 1
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

        self.Lstm_model = self.create_lstmmodel()
        self.Lstm_model.fit(X_train, y_train, epochs=10, verbose=1)

        pre = self.Lstm_model.predict(X_test)
        plt.plot(test_data[:-self.n_steps],'r',label = 'true')
        plt.plot(pre,'b',label = 'pre')
        plt.legend()
        plt.xlabel('Observation')
        plt.ylabel('consumption')
        plt.show()

        # pred = scaler.inverse_transform(pre)
        # tru = scaler.inverse_transform(test[n_steps:])




    def LSTM_predict(self):
        pass


    def fit(self, train_data):
        self.LSTM_train(train_data)
        return


    def predict(self, test_data):
        return


'''
################## 聚类代码
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
data = pd.read_excel('1001.csv', encoding='gbk')
for f in data:  # 插值法填充
    data[f] = data[f].interpolate()
    # data.dropna(inplace=True)
print(data['D2'][301])
def getPCAData(data, comp):
    pcaClf = PCA(n_components=comp, whiten=True)
    pcaClf.fit(data)
    data_PCA = pcaClf.transform(data)  # 用来降低维度
    return data_PCA
def modiData(data):
    x1 = []
    x2=[]
    for i in range(0,len(data+1)):
        x1.append(data[i][0])
        x2.append(data[i][1])
    x1=np.array(x1)
    x2=np.array(x2)
    #重塑数据
    X=np.array(list(zip(x1,x2))).reshape(len(x1),2)
    return X
#绘制样式
def drawKmodel(XData,t):
    plt.figure(figsize=(10,10))
    colors = ['g','r','y','b','c']
    markers = ['o','s','d','h','h']
    kmeans_model = KMeans(n_clusters=t).fit(XData)
    for i,l in enumerate(kmeans_model.labels_):
        plt.plot(XData[i][0],XData[i][1],color=colors[l],marker=markers[l],ls='None')
        plt.title('%s Countries K-Means'%(len(XData)))
    plt.show()
dataPCA = getPCAData(data,2)
dataX = modiData(dataPCA)
drawKmodel(dataX,5)
print(dataPCA.shape)
print(dataX.shape)

'''
