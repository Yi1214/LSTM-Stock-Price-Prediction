# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:42:51 2018

@author: ZhangYi
"""

"""
本程序适用场景：
LSTM
输入5min，输出1min，Moving 训练和测试
Example:
输入：1-5 2-6 3-7 4-8  5-9 ……
输出： 6   7   8   9   10  ……
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense, Activation
import numpy as np
import tensorflow as tf

np.random.seed(1337)  

#Load data
def load_data(stock):
    #train data path 
    spath=path+str(stock)+".csv"
    data=pd.read_csv(spath)
    return data

#Get Training data:
def Get_training_testing_data(data, period):
    data_set = np.array([data.iloc[:, 5].values,data.iloc[:, 8].values]).T
    training_number = len(data_set[:,0]) - 5543
    training_X = np.reshape(data_set[0:training_number, 0], (len(data_set[0:training_number, 0]), 1))
    training_Y = np.reshape(data_set[0:training_number, 1], (len(data_set[0:training_number, 1]), 1))
    testing_X = np.reshape(data_set[training_number:, 0], (len(data_set[training_number:, 0]), 1))
    testing_Y = np.reshape(data_set[training_number:, 1], (len(data_set[training_number:, 1]), 1))
    #Normalization
    training_X_1 = Normalization(training_X)
    training_Y_1 = Normalization(training_Y)
    testing_X_1 = Normalization(testing_X)
    testing_Y_1 = Normalization(testing_Y)
    X_train = np.zeros([len(training_X_1) - period ,period])
    for i in range(np.shape(X_train)[0]):
        X_train[i,:] = np.array(training_X_1[i:i + period]).T  
    X_test = np.zeros([len(testing_X_1) - period ,period])
    for i in range(np.shape(X_test)[0]):
        X_test[i,:] = np.array(testing_X_1[i:i + period]).T
    Y_train = np.reshape(training_Y_1[period:], (len(training_Y_1[period:]), 1))
    Y_test = np.reshape(testing_Y_1[period:], (len(testing_Y_1[period:]), 1)) 
    return X_train, X_test, Y_train, Y_test, testing_Y, testing_X #Output original testing_Y_1 in order to DeNormalize

#Normalize data
def Normalization(X):
    return [(x - X[0]) / X[0] for x in X]

#DeNormalize data
def DeNormalization(X, base):
    return [(x + 1) * base for x in X]

#Change all data to np.float32
def Change_form(X_train, X_test, Y_train, Y_test):
    #After De/Normalization, we must change data from list to array
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    #Before input into RNN, we must change the form of data to np.float32
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.float32)
    #In LSTM, X must be 3D type!!!
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_train, X_test, Y_train, Y_test

def Build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
    print(model.layers)
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(output_dim=1,use_bias=True))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model

def train_model(model, X_train, Y_train, X_test, Y_test):
    model = model
    model.fit(X_train, Y_train, batch_size=512, nb_epoch=200, validation_split=0)
    Y_pred = model.predict(X_test)
    Y_pred = np.reshape(Y_pred, (Y_pred.size, ))
    return Y_pred, Y_test

def Results_dispaly(stock, Y_test, Y_pred, X_test):
    Y_pred = Y_pred + X_test[period,0] - Y_pred[0,0]
    plt.xlabel('Time / min')
    plt.ylabel('Closing Price / RMB')
    plt.title("Prediction "+ str(stock))
    plt.plot(Y_pred,label='Prediction')
    plt.plot(Y_test,label='True')
    plt.legend(loc='upper right')
    RMSE = np.sqrt(np.mean((Y_test-Y_pred)*(Y_test-Y_pred)))
    print('The RMSE of ' + str(stock) + ' is: ' + str(RMSE))
    return Y_test, Y_pred
    
#Main part:
path="C:\\Information Theory\\Project\\Train\\"  
stock = 600064
data = load_data(stock)
period = 5
X_train, X_test, Y_train, Y_test, Y_test_org, X_test_org = Get_training_testing_data(data, period)
X_train, X_test, Y_train, Y_test = Change_form(X_train, X_test, Y_train, Y_test)
model = Build_model()
Y_pred, Y_test= train_model(model, X_train, Y_train, X_test, Y_test)
Y_test = np.array(DeNormalization(Y_test, Y_test_org[0]))
Y_pred = np.array(DeNormalization(Y_pred, Y_test_org[0]))
Results_dispaly(stock, Y_test, Y_pred, X_test_org)
