#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 23:23:22 2019

@author: anoop
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#sdsds


def plotprediction(test,predicted,name):
    plt.plot(test,color='red',label='Real Stock Prize of {}'.format(name))
    plt.plot(predicted,color='blue',label='Predicted Stock Prize of {}'.format(name))
    plt.title('Stock Prize of {}'.format(name))
    plt.xlabel('Time')
    plt.ylabel('Stock prize')
    plt.legend()
    plt.show()

def RMSE(test,predicted):
    rmse=math.sqrt(mean_squared_error(test,predicted))
    print('The RMSE is {}.'.format(rmse))
    
def transform(train):
    Xtrain=[]
    ytrain=[]
    for i in range(60,len(train)):
        Xtrain.append(train[i-60:i,0])
        ytrain.append(train[i,0])
    Xtrain=np.array(Xtrain)
    ytrain=np.array(ytrain)
    Xtrain=np.reshape(Xtrain,(Xtrain.shape[0],Xtrain.shape[1],1))
    return Xtrain,ytrain

def testValues(dataset,len_test):
    dataset_total = pd.concat((dataset["High"][:'2016'],dataset["High"]['2017':]),axis=0)
    inputs = dataset_total[len(dataset_total)-len_test - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs=sc.transform(inputs)
    X_test = []
    for i in range(60,311):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    return X_test


def LSTMmodel(Xtrain,ytrain,Xtest):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=50,return_sequences=True,input_shape=(Xtrain.shape[1],1)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50,return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50,return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer='rmsprop',loss='mean_squared_error')
    model.fit(Xtrain,ytrain,epochs=20,batch_size=32)
    
    result=model.predict(Xtest)
    return result

def GRUmodel(Xtrain,ytrain,Xtest):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(units=50,return_sequences=True,input_shape=(Xtrain.shape[1],1),activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.GRU(units=50,return_sequences=True,input_shape=(Xtrain.shape[1],1),activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.GRU(units=50,return_sequences=True,input_shape=(Xtrain.shape[1],1),activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.GRU(units=50,activation='tanh'))    
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
    model.fit(Xtrain,ytrain,epochs=20,batch_size=150)
    result=model.predict(Xtest)
    return result
            

amazon=pd.read_csv('/home/anoop/Downloads/Stock/IBM_2006-01-01_to_2018-01-01.csv',index_col='Date',parse_dates=['Date'])

amazon_train = amazon[:'2016'].iloc[:,1:2].values
amazon_test = amazon['2017':].iloc[:,1:2].values
sc=MinMaxScaler(feature_range=(0,1)) 
amazon_train=sc.fit_transform(amazon_train)
amazon_test=sc.fit_transform(amazon_test)
amazon_Xtrain,amazon_ytrain= transform(amazon_train)

amazon_Xtest=testValues(amazon,len(amazon_test))
amazon_predicted_lstm=LSTMmodel(amazon_Xtrain,amazon_ytrain,amazon_Xtest)
p=GRUmodel(amazon_Xtrain,amazon_ytrain,amazon_Xtest)

amazon_predicted_lstm=sc.inverse_transform(amazon_predicted_lstm)
p=sc.inverse_transform(p)
amazon_test1=sc.inverse_transform(amazon_test)

plotprediction(amazon_test1,amazon_predicted_lstm,"AMAZON")
RMSE(amazon_test,amazon_predicted_lstm)

plotprediction(amazon_test1,p,"AMAZON")











