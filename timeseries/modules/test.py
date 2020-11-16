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
np.arange(start,end,step = dt.timedelta(days=1))
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



#%%%
###### EXAMPLE PREPARE DATA LSTM ###

import numpy as np
import datetime
# load...
data = list()
n = 5000
for i in range(n):
	data.append([i+1, (i+1)*10])
data = np.array(data)
print(data[:5, :])
print(data.shape) 

# drop time
data = data[:, 1]
print(data.shape)
samples = list()
length = 200
# step over the 5,000 in jumps of 200
for i in range(0,n,length):
	# grab from i to i + 200
	sample = data[i:i+length]
	samples.append(sample)
print(len(samples))
# convert list of arrays into 2d array
data = np.array(samples)
print(data.shape)
# reshape into [samples, timesteps, features]
# expect [25, 200, 1]
data = data.reshape((len(samples), length, 1))
print(data.shape)



lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(200, return_sequences=False),
    tf.keras.layers.Dropout(.2),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(data, epochs=30, batch_size=20, verbose=1)

df.index
orig = np.array(df[df.columns[1]])
print(orig[:5])


# check if missing values 
time_seq = np.arange(datetime.date(2017,1,1),datetime.date(2019,3,13), datetime.timedelta(days=1))
for seq in time_seq:
    if seq not in df.index:
        print(seq)

#%%
## Original
# values_1 = df.values


## full data without resampling
#values = df.values

# integer encode direction
# ensure all data is float
#values = values.astype('float32')
# normalize features
scaler_1 = MinMaxScaler(feature_range=(0, 1))
print(values_1.shape)

scaled_1 = scaler_1.fit_transform(values_1)

new_df = pd.DataFrame(scaled_1, columns = df.columns)
# frame as supervised learning
reframed_1 = series_to_supervised(new_df, 1, 1)

# drop columns we don't want to predict
# reframed_1.drop(reframed_1.columns[[6,7,8,9]], axis=1, inplace=True)
print(reframed_1.head())

values_2 = reframed_1.values
print(values_2.shape)

n_train_time = int(df.shape[0]*.7)
n_val_time = int(df.shape[0]*.9)

train_1 = values_2[:n_train_time, :]
val_1 = values_2[n_train_time:n_val_time, :]
test_1 = values_2[n_val_time:, :]
print(train_1.shape, val_1.shape, test_1.shape) 

##test = values[n_train_time:n_test_time, :]
# split into input and outputs
train_X_1, train_y_1 = train_1[:, :-1], train_1[:, -1]
test_X_1, test_y_1 = test_1[:, :-1], test_1[:, -1]
val_X_1, val_y_1 = val_1[:, :-1], val_1[:, -1]
print(train_X_1.shape, train_y_1.shape, val_X_1.shape, val_y_1.shape,  test_X_1.shape, test_y_1.shape ) 

# reshape input to be 3D [samples, timesteps, features]
'''if more steps probably need to shift um die steps'''
train_X_res_1 = train_X_1.reshape((train_X_1.shape[0], 1, train_X_1.shape[1]))
test_X_res_1 = test_X_1.reshape((test_X_1.shape[0], 1, test_X_1.shape[1]))
val_X_res_1 = val_X_1.reshape((val_X_1.shape[0], 1, val_X_1.shape[1]))
print(train_X_res_1.shape, train_y_1.shape, val_X_res_1.shape, val_y_1.shape,  test_X_res_1.shape, test_y_1.shape ) 



lstm_model_1 = cerate_lstm(layers_count = 1, batch_size_list=[100], dropout=.2, multisteps = 1)
history = compile_and_fit(model = lstm_model_1, epochs = 40, X_train = train_X_res_1, 
                y_train = train_y_1, X_val = val_X_res_1, y_val= val_y_1, verbose = 1)

yhat_1 = lstm_model_1.predict(test_X_res_1)
test_X_res_res_1 = test_X_res_1.reshape((test_X_res_1.shape[0], 5))
print(yhat_1.shape, test_X_res_res_1.shape)

# invert scaling for forecast
inv_yhat_1 = np.concatenate((yhat_1, test_X_res_res_1[:, -4:]), axis=1)
print(inv_yhat_1.shape) # concated to -1 as the original

inv_yhat_2 = scaler_1.inverse_transform(inv_yhat_1)
print(inv_yhat_2.shape)

inv_yhat_3 = inv_yhat_2[:,0]
print(inv_yhat_3.shape)
# invert scaling for actual
test_y_res = test_y_1.reshape((len(test_y_1), 1))
inv_y_1 = np.concatenate((test_y_res, test_X_res_res_1[:, -4:]), axis=1)
print(test_y_res.shape, inv_y_1.shape)
inv_y_2 = scaler_1.inverse_transform(inv_y_1)
print(inv_y_2.shape)
inv_y_3 = inv_y_2[:,0]
print(inv_y_3.shape)
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y_3, inv_yhat_3))
print('Test RMSE: %.3f' % rmse)




