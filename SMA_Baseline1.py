# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 19:02:53 2018

@author: Alex Fang
"""

import pandas
import numpy as np
import os
import matplotlib.pyplot as plt

#train data path
path="C:\\Information Theory\\Project\\Train\\"
filelist=os.listdir(path) #返回指定路径中所有文件和文件夹的名字，并存在一个list中
#get stock list
#for any given stock, analyze it with input stock id
#每一次的input都是这一分钟的“Last Closing Price”以及之前
#每一次的output都是这一分钟的Closing Price
#假设回测浦发银行，某一天，5min为一个周期

#移动平均，后面再补充一个指数平均作为baseline2
def Baseline1(stock, period):
    spath = path+stock+".csv"
    data = pandas.read_csv(spath)
    data_get_1 = list(data.iloc[:, 5].values)
    data_get_2 = list(data.iloc[:, 8].values)
    Length = len(data_get_1)
    data_set_1 = data_get_1[Length-5543:Length]
    data_set_2 = data_get_2[Length-5543:Length]
    Input = data_set_1[:]
    Answer = data_set_2[period:] #舍弃前5min的数据，因为已用来预测。预测值为第6分钟以后，故需要从第6分钟开始比较
    Output=[]
    for i in range(5,len(Input)):
        Output.append(np.mean(Input[i-period:i]))
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

stock='600064'
period=5   
Y_pred, Y_true = Baseline1(stock,period)
Y_pred = np.array(Y_pred)
Y_true = np.array(Y_true)
RMSE = np.sqrt(np.mean((Y_true-Y_pred)*(Y_true-Y_pred)))
print('The RMSE of ' + str(stock) + ' is: ' + str(RMSE))