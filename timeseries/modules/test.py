#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:49:55 2020

@author: Kostja
"""

import plotly
import plotly.graph_objs as go

import plotly.express as px
df = einzel_aut
#fig = px.scatter_3d(df, x='Verkaufsdatum', y='Einzel Wert in EUR', z='Einzel Menge in ST')
fig = px.histogram(df, x="Einzel Wert in EUR") # histogram
fig = px.histogram(df, x="Einzel Menge in ST")
plotly.offline.plot(fig, filename = 'test')

    
trace0 = go.Scatter(
    x = einzel_aut['Verkaufsdatum'],
    y = einzel_aut['Einzel Menge in ST'],
    mode = 'lines',
    name = 'Einzel Menge'
    )
trace1 = go.Scatter(
    x = einzel_aut['Verkaufsdatum'],
    y = einzel_aut['Einzel Wert in EUR'],
    mode = 'lines',
    name = 'Einzel Wert'
    )
trace2 = go.Scatter(
    x = einzel_aut['Verkaufsdatum'],
    y = einzel_aut['4Fahrt Menge in ST'],
    mode = 'lines',
    name = '4Fahrt Menge'
    )
trace3 = go.Scatter(
    x = einzel_aut['Verkaufsdatum'],
    y = einzel_aut['4Fahrt Wert in EUR'],
    mode = 'lines',
    name = '4Fahrt Wert'
    )

plt = [trace0, trace1,trace2,trace3]
plotly.offline.plot(plt, filename = 'test')
    

trace0 = go.Contour(
    z = [einzel_aut['Verkaufsdatum'],einzel_aut['Einzel Wert in EUR'],einzel_aut['Einzel Menge in ST']],
    line = dict(smoothing=0.65),
    )
trace1 = go.Contour(
    z = [einzel_aut['Verkaufsdatum'],einzel_aut['4Fahrt Wert in EUR'],einzel_aut['4Fahrt Menge in ST']],
    line = dict(smoothing=0.8),
    )
plt = [trace0,trace1]

plotly.offline.plot(plt, filename = 'test')



trace0 = go.Bar(
    y = einzel_aut['Einzel Wert in EUR']
    )
trace1 = go.Bar(
    y = einzel_aut['Einzel Menge in ST']
    )
plt = [trace0,trace1]

plotly.offline.plot(plt, filename = 'test')


trace0 = go.Surface(z=einzel_aut.values)

plt = [trace0]

plotly.offline.plot(plt, filename = 'test')



def check_for_nan(data):
    
    result={}
    for nr, df in enumerate(data):
        if df.isnull().values.any():
            result.update({nr:True})
    
    if not result:
        result = False
    
    return result

if check_for_nan(data):
    print('The DataFrame has Nan´s in it, with following form {nr_datafarme, True for Nans}: ' + str(check_for_nan(data)))
    
#%%
    
#Importing the libraries
from nsepy import get_history as gh
import datetime as dt
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from pmdarima import auto_arima 
import warnings 
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.statespace.sarimax import SARIMAX
#Setting start and end dates and fetching the historical data
start = dt.datetime(2013,1,1)
end = dt.datetime(2019,12,31)
stk_data = gh(symbol='SBIN',start=start,end=end)


#Data Preprocessing
stk_data['Date'] = stk_data.index
data2 = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
data2['Date'] = stk_data['Date']
data2['Open'] = stk_data['Open']
data2['High'] = stk_data['High']
data2['Low'] = stk_data['Low']
data2['Close'] = stk_data['Close']



#####################ARIMA###############################
# Ignore harmless warnings 
warnings.filterwarnings("ignore") 

# Fit auto_arima function to Stock Market Data
stepwise_fit = auto_arima(data2['Close'], start_p = 1, start_q = 1, max_p = 3, max_q = 3, m = 12, start_P = 0, seasonal = True, d = None, D = 1, trace = True, error_action ='ignore', suppress_warnings = True, stepwise = True)         

  
# To print the summary 
stepwise_fit.summary() 

# Split data into train / test sets 
train = data2.iloc[:len(data2)-150] 
test = data2.iloc[len(data2)-150:]

# Fit a SARIMAX
model = SARIMAX(data2['Close'],  order = (0, 1, 1),  seasonal_order =(2, 1, 1, 12)) 


result = model.fit() 
result.summary() 


start = len(train) 
end = len(train) + len(test) - 1

  
# Predictions for one-year against the test set 
predictions = result.predict(start, end, typ = 'levels').rename("Predictions") 

  
# plot predictions and actual values 
predictions.plot(legend = True) 
test['Close'].plot(legend = True)


train_set = data2.iloc[0:1333:, 1:2].values
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(train_set)
X_train = []
y_train = []
for i in range(60, 1333):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0]) 
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#Defining the LSTM Recurrent Model
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))


#Compiling and fitting the model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 15, batch_size = 32)


#Fetching the test data and preprocessing
testdataframe = gh(symbol='SBIN',start=dt.datetime(2018,5,23),end=dt.datetime(2018,12,31))
testdataframe['Date'] = testdataframe.index
testdata = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
testdata['Date'] = testdataframe['Date']
testdata['Open'] = testdataframe['Open']
testdata['High'] = testdataframe['High']
testdata['Low'] = testdataframe['Low']
testdata['Close'] = testdataframe['Close']
real_stock_price = testdata.iloc[:, 1:2].values
dataset_total = pd.concat((data2['Open'], testdata['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(testdata) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 235):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


#Making predictions on the test data
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#Visualizing the prediction
plt.figure()
plt.plot(real_stock_price, color = 'r', label = 'Close')
plt.plot(predicted_stock_price, color = 'b', label = 'Prediction')
plt.xlabel('Date')
plt.legend()
plt.show()
#%%


splited_data = split(df, 1, 0.7)

trainX = splited_data['Train_X']
trainY = splited_data['Train_y']


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(4, return_sequences=True, input_shape =(trainX.shape[0],1)),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])





#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# data.reshape(-1,1) same as data.reshape((1, 10, 1))
data = data.reshape(-1,1) # -1 means we do not not the dim of rows in this example


def window_data_1(data, window_size):
    X = []
    y = []

        
    i = 0
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        
        i +=1
    assert len(X) == len(y)
    return X,y

scaler_1 = StandardScaler()
scaler_1 = scaler_1.fit(data)
data = scaler_1.transform(data)
X_1,y_1 = window_data_1(data, 7)

# split in train and test
X_train_1 = np.array(X_1[:int(len(X_1)*.7)])
y_train_1 = np.array(y_1[:int(len(X_1)*.7)])
X_val_1 = np.array(X_1[int(len(X_1)*.7):int(len(X_1)*.9)])
y_val_1 = np.array(y_1[int(len(X_1)*.7):int(len(X_1)*.9)])
X_test_1 = np.array(X_1[int(len(X_1)*.9):])
y_test_1 = np.array(y_1[int(len(X_1)*.9):])



model_1 = keras.Sequential()
model_1.add(layers.LSTM(8, return_sequences=True))
model_1.add(layers.Dense(1))
model_1(X_train_1)
model_1.summary()

model_1.compile(loss='mean_squared_error', optimizer='adam')
model_1.fit(X_train_1, y_train_1, epochs=30, batch_size=20, verbose=1, validation_data=(X_val_1,y_val_1) )

X_inv_trans_1 = scaler.inverse_transform(X_test_1)
y_inv_trans_1 = scaler.inverse_transform(y_test_1)

testPredict = model_1.predict(X_inv_trans_1, batch_size=20)
scaler.inverse_transform(testPredict)

score = model_1.evaluate(X_inv_trans_1, y_inv_trans_1, batch_size=20, verbose=1)

np.sqrt(score)

dat = list()
n = 5000
for i in range(n):
	dat.append([i+1.0])
dat = np.array(dat)

# create window
def window_data(data, window_size, shift, label_size):
    X = []
    y = []
    

    for i in np.arange(len(data)-1, step = 1):
        if i+window_size >= len(data)-1:
            break
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size + label_size])
        
    # i = 0
    # while (i + window_size) <= len(data) - 1:
    #     X.append(data[i:i+window_size])
    #     y.append(data[i+window_size])
        
    #     i +=1
    assert len(X) == len(y)
    return X,y

scaler = StandardScaler()
scaler = scaler.fit(dat)
dat = scaler.transform(dat)
X,y = window_data(dat, 7, 1, 1)

X_train = np.array(X[:int(len(X)*.7)])
y_train = np.array(y[:int(len(X)*.7)])
X_val = np.array(X[int(len(X)*.7):int(len(X)*.9)])
y_val = np.array(y[int(len(X)*.7):int(len(X)*.9)])
X_test = np.array(X[int(len(X)*.9):])
y_test = np.array(y[int(len(X)*.9):])


model = keras.Sequential()
model.add(layers.LSTM(8, return_sequences=True))
model.add(layers.Dense(1))
model(X_train)
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=30, batch_size=20, verbose=1, validation_data=(X_val,y_val))

prediction = model.predict(X_test, batch_size=20)
pred = scaler.inverse_transform(prediction)

X_inv_trans = scaler.inverse_transform(X_test)
y_inv_trans = scaler.inverse_transform(y_test)

mse = model.evaluate(X_inv_trans, y_inv_trans, batch_size=20, verbose=1)
np.sqrt(mse)

y_inv_trans

plt.plot(X_inv_trans[:,5,:])
plt.plot(pred[:,5,:])

X_inv_trans[:,6,:]
X_inv_trans.shape
pred = pred[0,:,0]


orig = scaler.inverse_transform(y_test)
orig = orig[:,0,0]

plt.plot(orig, label='orig')
plt.plot(pred, label='pred')
plt.legend(loc='best')



plt.plot(X_test[0], label = 'test')
plt.plot(y_test[0], label = 'y_test')
plt.legend(loc='best')
#%%




lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.Dropout(.2),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

lstm_model(X_train_1)
lstm_model.summary()


lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(X_train, y_train, epochs=30, batch_size=20, verbose=1)

prediction = lstm_model.predict(X_test)
mean_squared_error(y_test, prediction)

tf.compat.v1.placeholder(tf.float32, shape=(1024, 1024))
inputs = tf.placeholder(tf.float32, [32,7,1])
targets = tf.placeholder(tf.float32, [32,1])


#%%

# load...

