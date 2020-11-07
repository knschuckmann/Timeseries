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
    
import os
import datetime

# import IPython
# import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import tensorflow as tf   
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error


from timeseries.modules.config import ORIG_DATA_PATH, SAVE_PLOTS_PATH, SAVE_MODELS_PATH, DATA, SAVE_RESULTS_PATH, ORIGINAL_PATH
from timeseries.modules.dummy_plots_for_theory import save_fig, set_working_directory
from timeseries.modules.load_transform_data import load_transform_excel
from timeseries.modules.baseline_prediction import combine_dataframe, monthly_aggregate, split

def cerate_lstm(layers_count=2, batch_size_list=[64, 64], dropout=0.2, **kwargs):
    
    model = tf.keras.Sequential()
    
    batch_size_list = [3]
    for number in range(len(batch_size_list) -1):
        model.add(tf.keras.layers.LSTM(batch_size_list[number], return_sequences=True ,**kwargs))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
    
    model.add(tf.keras.layers.LSTM(batch_size_list[-1], return_sequences=False,**kwargs))
    if dropout > 0:
        model.add(tf.keras.layers.Dropout(dropout))
    
    model.add(tf.keras.layers.Dense(units = 2))
    
    return model

def compile_and_fit(model, epochs, X_train,y_train, X_val, y_val, patience = 3, **kwargs):        
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    return model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),
                        callbacks=[early_stopping], **kwargs)
    
def set_index_in_df_list(dataframe):
    for nr, data in enumerate(dataframe):
        data_frame[nr] = data.set_index(data['Verkaufsdatum'], drop = True)
    return dataframe 


def remove_unimportant_columns(all_columns, column_list):
    
    result_columns = set(all_columns)
    for column in column_list:
        try:
            result_columns -= set([column])
        except:
            continue
    return result_columns
 

def train_val_test(data, rel_train, rel_val):
    length = len(data)
    size_train = int(length*rel_train)
    size_test = int(length*rel_val)    
    # train, val, test
    return data[:size_train], data[size_train:size_test], data[size_test:]
    

def scale_data_to_standrad(scale_method, train, test, val):
    # fit scaler
    scaler = scale_method
    scaler = scaler.fit(train)
    # transform train
    train_set = scaler.transform(train)
    test_set = scaler.transform(test)
    val_set = scaler.transform(val)
    
    return scaler, train_set, test_set, val_set

def check_if_column_right_place(column, dataframe):
    
    first_series  = dataframe[dataframe.columns[np.where(dataframe.columns == column)]]
    other_series = dataframe[dataframe.columns[np.where(dataframe.columns != column)]]
    
    return first_series.merge(other_series, left_on='date', right_on= 'date')

def reshape_array(numpy_array, timesteps):
    
    new_list = [numpy_array[x:x+timesteps] for x in range(int(np.ceil(numpy_array.shape[0]/timesteps)))]
    return np.array(new_list)

def series_to_supervised(data_frame, n_in=1, n_out=1, dropnan=True):
    
    n_vars = 1 if type(data_frame) is list else data_frame.shape[1]
    names_list = data_frame.columns
    cols, names = list(), list()
	# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data_frame.shift(i))
        names += [(names_list[j] + '(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data_frame[names_list[0]].shift(-i))
        if i == 0:
            names.append(names_list[0] + '(t)')
        else:
            names.append(names_list[0] + '(t+%d)' % (i))
	# put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
	# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


set_working_directory()
data_frame = load_transform_excel(ORIG_DATA_PATH)
events = pd.read_excel(ORIGINAL_PATH + 'events.xlsx')
try:
    ran
except:
    combined_m_df, combined_df = combine_dataframe(data_frame, monthly = True)
    monthly_list = list()
    for df in data_frame:
        monthly_list .append(monthly_aggregate(df, combined = True))
    data_frame = set_index_in_df_list(data_frame)
    ran = True 


features = 2
timesteps = 2
time_vars = timesteps *df.shape[1]
feat_vars = features *df.shape[1]
max_vars = time_vars + feat_vars
take_column = 'Einzel Menge in ST'

# set the DataFrame to predict 
data_orig = pd.DataFrame(combined_df[take_column], columns = [take_column])

data_orig = data_orig.merge(events, left_on =data_orig.index, right_on='date')
data_orig = data_orig.set_index(data_orig['date'], drop = True, verify_integrity= True)
data_orig = data_orig.drop('date', axis = 1)

used_columns = remove_unimportant_columns(data_orig.columns, ['Verkaufsdatum','Tages Wert in EUR','Einzel Wert in EUR','4Fahrt Wert in EUR', 'Gesamt Wert in EUR'])
df = data_orig[used_columns] 
df = check_if_column_right_place(column = take_column, dataframe = df)   

reframed = series_to_supervised(df, n_in = features, n_out = timesteps)


# Split data 
train_df, val_df, test_df = train_val_test(reframed, rel_train = .7, rel_val = .9)

# scale data
scaler, sc_train, sc_test, sc_val = scale_data_to_standrad(MinMaxScaler(feature_range=(0, 1)), train_df, test_df, val_df)
# sc_train_df = pd.DataFrame(sc_train, columns=train_df.columns, index = train_df.index)
# sc_test_df = pd.DataFrame(sc_test, columns=test_df.columns, index = test_df.index)
# sc_val_df = pd.DataFrame(sc_val, columns=val_df.columns, index = val_df.index)

train_df.shape
# split into input and outputs
train_X, train_y = sc_train[:, :-timesteps], sc_train[:, -timesteps:]
val_X, val_y = sc_val[:, :-timesteps], sc_val[:, -timesteps:]
test_X, test_y = sc_test[:, :-timesteps], sc_test[:, -timesteps:]
print(train_X.shape, train_y.shape, val_X.shape, val_y.shape,  test_X.shape, test_y.shape)


'''
sequence_length: Length of the output sequences (in number of timesteps).
batch_size: Number of timeseries samples in each batch (except maybe the last one).
'''
new = tf.keras.preprocessing.timeseries_dataset_from_array(data = train_X, targets = train_y, 
                                                           sequence_length = timesteps, batch_size=16)

new_val = tf.keras.preprocessing.timeseries_dataset_from_array(data = val_X, targets = val_y, 
                                                           sequence_length = timesteps, batch_size=16)


new_test = tf.keras.preprocessing.timeseries_dataset_from_array(data = test_X, targets = test_y, 
                                                           sequence_length = timesteps, batch_size=16)

for i in new_test:
    print('i_0: ',i[0].shape)
    print('i_1: ',i[1].shape)


train_X = reshape_array(train_X, timesteps)
val_X = reshape_array(val_X, timesteps)
test_X = reshape_array(test_X, timesteps)

train_y = reshape_array(train_y, timesteps)
val_y = reshape_array(val_y, timesteps)
test_y = reshape_array(test_y, timesteps)
print(train_X.shape, train_y.shape, val_X.shape, val_y.shape,  test_X.shape, test_y.shape )

# print(train_X.shape, train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
# train_X = train_X.reshape(train_X.shape[0], timesteps, train_X.shape[1])
# val_X = val_X.reshape(val_X.shape[0], timesteps, val_X.shape[1])
# test_X = test_X.reshape(test_X.shape[0], timesteps, test_X.shape[1])

# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].
lstm_model = cerate_lstm(layers_count = 1, batch_size_list=[100], dropout=.2)
# history = compile_and_fit(model = lstm_model, epochs = 40, X_train = train_X, 
#                 y_train = train_y, X_val = val_X, y_val= val_y, verbose = 1, batch_size = timesteps)


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=3,
                                                      mode='min')
lstm_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanAbsoluteError()])

lstm_model.fit(new, epochs=40, validation_data=new_val,
                        callbacks=[early_stopping])

lstm_model.evaluate(new_test)
yhat = lstm_model.predict(new_test)

inputs, targets = tf.data.experimental.get_single_element(new_test.take(1))



tf.keras.preprocessing.sequence.pad_sequences(train_y)
inputs.shape
targets.shape

scaler.inverse_transform(yhat)
# make a prediction
# yhat = lstm_model.predict(test_X)
test_X = test_X[:-1]
test_X = test_X.reshape((test_X.shape[0], feat_vars))

yhat.shape
test_X.shape

inv_yhat = np.concatenate((yhat, test_X), axis=1)
inv_yhat.shape
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:,-feat_vars:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,:2]


pd.DataFrame(inv_yhat)

# invert scaling for actual values
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -feat_vars:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

plt.plot(inv_y)
test_y = new_test
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

performance = {}
performance['MSE'] = lstm_model.evaluate(new_test, verbose=1)
performance['RMSE'] = np.sqrt(performance['MSE'][0])

test_y_1 = test_y.reshape((len(test_y), 1))

test_y_1 = test_y[:-1]
test_X_1 = test_X

test_y_1 = test_y_1[:,1:2]
test_y_1.shape
inv_y = np.concatenate((test_y_1, test_X_1), axis=1)

sc_test.inverse_transform()
pd.DataFrame(inv_y)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,:2]

for i in range(2):
    rmse = np.sqrt(mean_squared_error(inv_y[:,i], inv_yhat[:,i]))
    print('Test RMSE: %.3f' % rmse)



#%%
# combined_df.shape
# values = combined_df.values 

df = df[[df.columns[4], df.columns[0], df.columns[1], df.columns[2], df.columns[3]]]

df.shape
values = df.values


scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = MinMaxScaler(feature_range=(0, 1))
print(values.shape)
print(values_1.shape)


# frame as supervised learning

features = 2
timesteps = 1
time_vars = timesteps *df.shape[1]
feat_vars = features *df.shape[1]
max_vars = time_vars + feat_vars



reframed_train = series_to_supervised(values,df.columns,  features, timesteps)
print(reframed_train.shape)
print(reframed_1.shape)

# reframed_val = series_to_supervised(scaled,sc_val_df.columns,  features, timesteps)
# reframed_test = series_to_supervised(scaled,sc_test_df.columns,  features, timesteps)
# # drop columns we don't want to predict
reframed_train.drop(reframed_train.columns[np.arange(feat_vars + 1, max_vars , 1)], axis=1, inplace=True)
# reframed_val.drop(reframed_val.columns[np.arange(feat_vars + 1,max_vars,1)], axis=1, inplace=True)
# reframed_test.drop(reframed_test.columns[np.arange(feat_vars + 1,max_vars,1)], axis=1, inplace=True)
print(reframed_train.shape)
print(reframed_1.shape)


scaled = scaler.fit_transform(reframed_train)
print(scaled.shape)
print(scaled_1.shape)


values_new = scaled

print(reframed_train.head())
print(reframed_1.head())
# split into train and test sets
# values_new = reframed_train.values
print(values_new.shape)
print(values_2.shape)

n_train_time = int(df.shape[0]*.7)
n_val_time = int(df.shape[0]*.9)


train, val, test = train_val_test(values_new,.7,.9)

print(train.shape, val.shape, test.shape) 
print(train_1.shape, val_1.shape, test_1.shape) 

##test = values[n_train_time:n_test_time, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
val_X, val_y = val[:, :-1], val[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
print(train_X.shape, train_y.shape, val_X.shape, val_y.shape,  test_X.shape, test_y.shape ) 
print(train_X_1.shape, train_y_1.shape, val_X_1.shape, val_y_1.shape,  test_X_1.shape, test_y_1.shape ) 

# reshape input to be 3D [samples, timesteps, features]
train_X_res = train_X.reshape(train_X.shape[0], timesteps, train_X.shape[1])
val_X_res = val_X.reshape(val_X.shape[0], timesteps, val_X.shape[1])
test_X_res = test_X.reshape(test_X.shape[0], timesteps, test_X.shape[1])

print(train_X_res.shape, train_y.shape, val_X_res.shape, val_y.shape,  test_X_res.shape, test_y.shape ) 
print(train_X_res_1.shape, train_y_1.shape, val_X_res_1.shape, val_y_1.shape,  test_X_res_1.shape, test_y_1.shape ) 


# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].
lstm_model = cerate_lstm(layers_count = 1, batch_size_list=[100], dropout=.2)
history = compile_and_fit(model = lstm_model, epochs = 40, X_train = train_X_res, 
                y_train = train_y, X_val = val_X_res, y_val= val_y, verbose = 1)


# lstm_model.save('model_hey')
# new = tf.keras.models.load_model('model_hey')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()



# make a prediction
yhat = lstm_model.predict(test_X_res)
test_X_res_res = test_X_res.reshape((test_X_res.shape[0], 10))
print(yhat.shape, test_X_res_res.shape)
print(yhat_1.shape, test_X_res_res_1.shape)
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X_res_res[:,-(10):]), axis=1)
print(inv_yhat.shape)
print(inv_yhat_1.shape)

inv_yhat_new = scaler.inverse_transform(inv_yhat)
print(inv_yhat_new.shape)
print(inv_yhat_2.shape)
inv_yhat_new_1 = inv_yhat_new[:,0]
print(inv_yhat_new_1.shape)
print(inv_yhat_3.shape)

# invert scaling for actual
test_y_new = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y_new, test_X_res_res[:, -(feat_vars):]), axis=1)
print(test_y_new.shape, inv_y.shape)
print(test_y_res.shape, inv_y_1.shape)

inv_y_new = scaler.inverse_transform(inv_y)
print(inv_y_new.shape)
print(inv_y_2.shape)

inv_y_new_1 = inv_y_new[:,0]
print(inv_y_new_1.shape)
print(inv_y_3.shape)
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y_new_1, inv_yhat_new_1))
print('Test RMSE: %.3f' % rmse)

# lstm_model.summary()
# plt.plot(inv_yhat, label = 'pred')
# plt.plot(inv_y,label = 'orig')
# plt.legend(loc = 'best')

#%%
## Original
values_1 = df.values


## full data without resampling
#values = df.values

# integer encode direction
# ensure all data is float
#values = values.astype('float32')
# normalize features
scaler_1 = MinMaxScaler(feature_range=(0, 1))
print(values_1.shape)

scaled_1 = scaler_1.fit_transform(values_1)
# frame as supervised learning
reframed_1 = series_to_supervised(scaled_1, df.columns, 1, 1)

# drop columns we don't want to predict
reframed_1.drop(reframed_1.columns[[6,7,8,9]], axis=1, inplace=True)
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



lstm_model_1 = cerate_lstm(layers_count = 1, batch_size_list=[100], dropout=.2)
history = compile_and_fit(model = lstm_model_1, epochs = 40, X_train = train_X_res_1, 
                y_train = train_y_1, X_val = val_X_res_1, y_val= val_y_1, verbose = 0)

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
inv_y_2 = scaler.inverse_transform(inv_y_1)
print(inv_y_2.shape)
inv_y_3 = inv_y_2[:,0]
print(inv_y_3.shape)
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y_3, inv_yhat_3))
print('Test RMSE: %.3f' % rmse)

#%%

## Univariate
values_1 = df['Einzel Menge in ST'].values


## full data without resampling
#values = df.values

# integer encode direction
# ensure all data is float
#values = values.astype('float32')
# normalize features
scaler_1 = MinMaxScaler(feature_range=(0, 1))
print(values_1.shape)


scaled_1 = scaler_1.fit_transform(values_1.reshape(-1,1))
# frame as supervised learning
reframed_1 = series_to_supervised(scaled_1, df.columns, 1, 1)

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



lstm_model_1 = cerate_lstm(layers_count = 1, batch_size_list=[100], dropout=.2)
history = compile_and_fit(model = lstm_model_1, epochs = 40, X_train = train_X_res_1, 
                y_train = train_y_1, X_val = val_X_res_1, y_val= val_y_1, verbose = 0)

yhat_1 = lstm_model_1.predict(test_X_res_1)
test_X_res_res_1 = test_X_res_1.reshape((test_X_res_1.shape[0], 1))
print(yhat_1.shape, test_X_res_res_1.shape)

# invert scaling for forecast
inv_yhat_1 = np.concatenate((yhat_1, test_X_res_res_1[:, :]), axis=1)
print(yhat_1.shape, inv_yhat_1.shape) # concated to -1 as the original

inv_yhat_2 = scaler_1.inverse_transform(inv_yhat_1)
print(inv_yhat_2.shape)

inv_yhat_3 = inv_yhat_2[:,0]
print(inv_yhat_3.shape)
# invert scaling for actual
test_y_res = test_y_1.reshape((len(test_y_1), 1))
inv_y_1 = np.concatenate((test_y_res, test_X_res_res_1[:,:]), axis=1)
print(test_y_res.shape, inv_y_1.shape)
inv_y_2 = scaler.inverse_transform(inv_y_1)
print(inv_y_2.shape)
inv_y_3 = inv_y_2[:,0]
print(inv_y_3.shape)
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y_3, inv_yhat_3))
print('Test RMSE: %.3f' % rmse)




