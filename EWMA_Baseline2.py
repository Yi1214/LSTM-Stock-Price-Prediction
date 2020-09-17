# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:43:36 2018

@author: ZhangYi
"""

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

#train data path
path="C:\\Information Theory\\Project\\Train\\"
filelist=os.listdir(path) #返回指定路径中所有文件和文件夹的名字，并存在一个list中


#移动平均，后面再补充一个指数平均作为baseline2
def Baseline2(stock, period, alpha):
    spath = path+stock+".csv"
    data = pd.read_csv(spath)
    data_get_1 = list(data.iloc[:, 5].values)
    data_get_2 = list(data.iloc[:, 8].values)
    Length = len(data_get_1)
    data_set_1 = data_get_1[Length-5543:Length]
    data_set_2 = data_get_2[Length-5543:Length]
    Input = data_set_1[:]
    Answer = data_set_2[period:] #舍弃前5min的数据，因为已用来预测。预测值为第6分钟以后，故需要从第6分钟开始比较
    
    Output=[]
    alpha = alpha
    
    x1_predict = np.mean(Input[0:period])
    x1 = Input[period]
    Output.append(alpha * x1 + (1-alpha) * x1_predict)
    for i in range(period + 1, len(data_set_1)):
        Output.append(alpha * Input[i] + (1 - alpha) * Output[i-period-1])
    #Analyze the performance
    plt.xlabel('Time / min')
    plt.ylabel('Close Price / RMB')
    x=range(len(Input)-period)
    y1=Output
    y2=Answer
    plt.title("Prediction "+ str(stock))
    plt.plot(x,y1,label='Prediction')
    plt.plot(x,y2,label='True')
    plt.legend(loc='upper right')
    return y1, y2

stock = '600064'
period = 5
Y_pred, Y_true = Baseline2(stock, period, 0.6)
Y_pred = np.array(Y_pred)
Y_true = np.array(Y_true)
RMSE = np.sqrt(np.mean((Y_true-Y_pred)*(Y_true-Y_pred)))
print('The RMSE of ' + str(stock) + ' is: ' + str(RMSE))