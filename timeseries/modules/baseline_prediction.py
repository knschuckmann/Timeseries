#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:57:33 2020

@author: Konstantin Schuckmann
"""


# A problem with ARIMA is that it does not support seasonal data. That is a time series with a repeating cycle.
# ARIMA expects data that is either not seasonal or has the seasonal component removed, e.g. seasonally adjusted via methods such as seasonal differencing.

from timeseries.modules.config import ORIG_DATA_PATH
from timeseries.modules.dummy_plots_for_theory import save_fig, set_working_directory
from timeseries.modules.load_transform_data import load_transform_excel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

from statsmodels.tsa.stattools import adfuller # dickey fuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error





def input_ar_ma(model_name):
    print('\nPlease insert the AR and the MA order which you can read from the PACF and ACF plots!\nIf you like to close the input, type in <stop>!')
    while True:
        try:
            nr = input('AR-order (PACF)\tMA-order (ACF):\n')
            if 'stop' in nr:
                break
            nr1, nr2 = nr.split()
            nr1, nr2 = int(nr1), int(nr2)
            if model_name in ['SARIMAX']:
                try:
                    nr_sari = input('Seasonal\nAR-order (PACF)\tMA-order (ACF)\tSeasonality:\n')
                    if 'stop' in nr_sari:
                        break
                    nr3, nr4, nr5 = nr_sari.split()
                    nr3, nr4, nr5 = int(nr3), int(nr4), int(nr5)
                    return {'AR':nr1, 'MA':nr2, 'SAR':nr3, 'SMA':nr4, 'S':nr5}
                    break
                except ValueError:
                    print('\nYou did not provide three numbers.\nPlease insert three numbers and no Strings!\nFormat: <nr> <nr> <nr>')
            else:
                return {'AR':nr1, 'MA':nr2}
                break
        except ValueError:
            print('\nYou did not provide two numbers.\nPlease insert two numbers and no Strings!\nFormat: <nr> <nr>')

def get_stationarity(timeseries, window, print_results = False):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=window).mean()
    rolling_std = timeseries.rolling(window=window).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation for windowsize ' + str(window) )
    plt.show(block = False)
    
    # Dickey–Fuller test:
    result = adfuller(timeseries)
    stationary = False
    
    if result[1] <= 0.05:
        stationary = True
    
    result = {'ADF Statistic':result[0], 'p-value':result[1], 'other_first': result[2], 
              'other_second':result[3], 'Critical Values':{'1%':result[4]['1%'],
                                                           '5%':result[4]['5%'],
                                                           '10%':result[4]['10%']},
              'stationary': stationary}
    if print_results:
        print('ADF Statistic: {}'.format(result['ADF Statistic']))
        print('p-value: {}'.format(result['p-value']))
        print('Critical Values:')
        for key, value in result['Critical Values'].items():
            print('\t{}: {}'.format(key, value))
        print('Stationary: {}'.format(stationary))
        
    return result

def plot_acf_pacf(series, rolling_window):
    if len(series.values)*0.5 >= 60: # big value for seasonal AR and MA detection
        lags_length = 60
    else:
        lags_length = len(series.values)*0.5 - 5 # -5 because otherwise lags_length to long to display
        
    fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
    fig = plot_pacf(series, zero = True, lags = lags_length, method = 'OLS', title ='Partial correlogram to find out AR value for ARMA window ' + str(rolling_window), ax = ax1, markerfacecolor='black', color = 'black')
    fig = plot_acf(series, zero = True, lags = lags_length, title ='Correlogram to find out MA value for ARMA window ' + str(rolling_window), ax = ax2, markerfacecolor='black', color = 'black')
    plt.show(block= False)
    
def decompose_and_plot(dependent_var, index, rolling_window, print_results = False):
    
    series = pd.Series(dependent_var.values, index=index)
    decomposition = seasonal_decompose(series)
    decomposition.resid.dropna(inplace=True)
    decomposition.plot()

    plot_acf_pacf(series, rolling_window)
    
    plt.figure(2)
    dickey_fuller_results = get_stationarity(decomposition.observed, window = rolling_window, print_results = print_results)

    return decomposition


def split(data, diff_faktor, rel_faktor):
    
    values = pd.DataFrame(data.values)
    dataframe = None
    
    if diff_faktor != 0:
        for i in range(1,diff_faktor+1):
            dataframe = pd.concat([dataframe,values.shift(i)], axis= 1)
        
        dataframe = pd.concat([dataframe, values], axis = 1)
    else:
        dataframe = values

    X = dataframe.values
    train, test = X[1:int(len(X)*rel_faktor)], X[int(len(X)*rel_faktor):]
    if diff_faktor != 0:
        train_X, train_y = train[:,:diff_faktor], train[:,diff_faktor]
        test_X, test_y = test[:,:diff_faktor], test[:,diff_faktor]
    else:
        train_X, train_y = None, train
        test_X, test_y = None, test
    
    return {'Train':train, 'Test':test, 'Train_X':train_X , 'Train_y':train_y , 'Test_X': test_X, 'Test_y':test_X}

def compare_models(given_model, data, diff_faktor, forecast_one_step = False):
  
    subresults = pd.DataFrame(columns = ['Predicted', 'Expected'])
    result = {'Used Model':None, 'MSE':None, 'RMSE':None}  
    decomposition = decompose_and_plot(dependent_var=data, index=data.index, rolling_window=rolling_window, print_results = True)
    
    if given_model in ['persistance','SARIMAX']:
        splitted_data = split(data = decomposition.observed, diff_faktor= 0, rel_faktor = 0.9 )
        if given_model == 'SARIMAX':
            order_dict = input_ar_ma(given_model)
            
    elif given_model in ['ARIMA', 'ARMA']: 
        diff_df = decomposition.observed.diff(diff_faktor)
        diff_df.dropna(inplace=True)
        print('\nAfter differencing:')
        get_stationarity(diff_df, window= rolling_window, print_results = True) # proove data is now stationary
        plot_acf_pacf(diff_df, diff_faktor)        
        splitted_data = split(data = diff_df, diff_faktor= 0, rel_faktor = 0.9 )
        order_dict = input_ar_ma(given_model)
    
    history = [x for x in splitted_data['Train']]    
    predictions = list()
    
    if forecast_one_step:
        test_length = 1
        splited_length = len(splitted_data['Test'])
    else:
        test_length = len(splitted_data['Test'])
        splited_length = 1
        
    for i in range(splited_length):
        # predict
        warnings.filterwarnings('ignore')
        if given_model == 'persistance':    
            yhat = history[-test_length:]
        elif given_model == 'ARIMA':
            model_fit = ARIMA(history, order=(order_dict['AR'],1,order_dict['MA'])).fit()
            yhat = model_fit.forecast(test_length)
        elif given_model == 'ARMA':
            model_fit = ARMA(history, order=(order_dict['AR'],order_dict['MA'])).fit(disp=0)
            yhat = model_fit.forecast(test_length)[0] 
        elif given_model == 'SARIMAX':
            model_fit = SARIMAX(history, order = (order_dict['AR'],1,order_dict['MA']), 
                                seasonal_order=(order_dict['SAR'],1,order_dict['SMA'],order_dict['S']), 
                                enforce_stationarity=False, enforce_invertibility = False).fit(disp = 0)
            yhat = model_fit.forecast(test_length)
       
        predictions.append(yhat)
        
        if forecast_one_step:
            # observation
            obs = splitted_data['Test'][i]
            history.append(obs)
            subresults.loc[i] = [yhat,obs]
        else:
            obs = splitted_data['Test']
            obs = [x for x in obs]
            subresults = [yhat,obs]
        
        print('>Predicted={}, Expected={}'.format(yhat, obs))
        
    result['Used Model'] = given_model
    if given_model == 'persistance':   
        predictions = sum(predictions, [])
    else:
        predictions = predictions[0]
    result['MSE'] = mean_squared_error(splitted_data['Test'][:,0], predictions, squared = True)
    result['RMSE'] = mean_squared_error(splitted_data['Test'][:,0], predictions, squared = False)
    print('RMSE: %.3f' % result['RMSE'])
    warnings.filterwarnings('default')

    return [result, subresults]


# res = result
# arma_res = result
# arima_res = result
# sarima_res = result       @


    

#%%

set_working_directory()
data_frame = load_transform_excel(ORIG_DATA_PATH)


def load_data():
    head_names = ['Einzel Menge in ST', '4Fahrt Menge in ST', 'Tages Menge in ST', 'Gesamt Menge in ST']
    df1 = data_frame[0]
    df1.index = df1['Verkaufsdatum']
    result_df = pd.DataFrame(columns = head_names)
    result_df[list(set(head_names) - set(['Tages Menge in ST']))] = df1[list(set(head_names) - set(['Tages Menge in ST']))]
    result_df['Tages Menge in ST'] = np.zeros(result_df.shape[0], dtype = int)
    
    for df in data_frame[1:]:
        df.index = df['Verkaufsdatum']
    
        temp_df = df
        for time_stamp in set(df1.index):
            if time_stamp not in set(df.index):
                dic = {dict_key : 0 for dict_key in df.columns[1:]}
                dic['Verkaufsdatum'] = time_stamp
                temp_df = temp_df.append(dic, ignore_index = True)
                
        temp_df.index = temp_df['Verkaufsdatum']
        
        for name in head_names:
            try:
                result_df[name] = result_df[name] + temp_df[name]
            except:
                print('This header is not present in temp',name)
                pass
    
    return result_df




pd.merge(df1, df2, on=['Verkaufsdatum', 'Verkaufsdatum']).sum(axis=1)
df1.merge()
neu = df1.add(2)
df2['Einzel Menge in ST'], index = df1.index)
neu = data_frame[0].merge(data_frame[1], on = 'Verkaufsdatum' )

data = data_frame[0]

to_nr = len(data)
rolling_window = 7
diff_faktor =7

# Plot data to get a first insight if it is stationary 
x = data['Verkaufsdatum'][:to_nr ]
y = data['Einzel Menge in ST'][:to_nr]
dickey_fuller_results = get_stationarity(data['Einzel Menge in ST'][:to_nr], window = rolling_window, print_results = True)
decomposition_non_stationary = decompose_and_plot(dependent_var=y, index=x, rolling_window=rolling_window)


new = np.log(data['Einzel Menge in ST'][:to_nr])
dec = decompose_and_plot(dependent_var=new, index=x_stationary, rolling_window=rolling_window )
hey = dec.observed - dec.seasonal

plt.plot(hey)
plt.plot(dec.resid)
# Difference Data to make it stationary
x_stationary = data['Verkaufsdatum'][diff_faktor:to_nr]
y_stationary = data['Einzel Menge in ST'][:to_nr].diff(diff_faktor).dropna()
# use decomposition function to find trend seasonality and resid
decomposition_orig = decompose_and_plot(dependent_var=y_stationary, index=x_stationary, rolling_window=rolling_window ,  print_results = True)

# because ARMA not good fot long tern therefore the split is 0.9 and not as usual 0.8
splited_data = split_dataset_for_training(x_stationary, y_stationary, .9)

decomposition = decompose_and_plot(dependent_var=splited_data['y_train'], index=splited_data['X_train'], rolling_window=rolling_window )
# We see from the plot that the best MA = 9 and best AR = 10 

warnings.filterwarnings('ignore')

# model = model_old

model = ARMA_and_plot( AR_value = 10, MA_value = 9 , orig_data = splited_data['y_train'],
              diff_faktor = diff_faktor, start = 610, end= 680, model = model)


rolling_mean = splited_data['y_train'].rolling(window=rolling_window).mean()
rolling_mean.dropna(inplace =True)
mse_mean = np.square(splited_data['y_train'][7:] - rolling_mean).mean()
mse_baseline = np.square(model.resid).mean() 


values = pd.DataFrame(y.values)

model.summary()





#%%

    


#%%
nr = len(splitted_data['Test'])

yhat = splitted_data['Train'][-nr:]
mean_squared_error(splitted_data['Test'], yhat, squared=False)

np.sqrt(np.square(- yhat.values).mean())

model_fit_arima = ARIMA(splitted_data['Train'], order=(4,1,3)).fit()
yhat_arima = model_fit_arima.forecast(1)[0]
np.sqrt(np.square(splitted_data['Test'] - yhat_arima).mean())
plot_acf(model_fit_arima.resid)
plot_pacf(model_fit_arima.resid)



data_diff = data.diff(1)
data_diff.dropna(inplace=True)
get_stationarity(timeseries = data_diff, window = 1)
splitted_data = split(data = data_diff, diff_faktor= 0, rel_faktor = 0.9 )
decomposition = decompose_and_plot(data_diff, data_diff.index,1,True)

model_fit_arma = ARMA(splitted_data['Train'], order=(8,3)).fit(disp=0)
yhat_arma = model_fit_arma.forecast(nr)[0] 
mean_squared_error(splitted_data['Test'],yhat_arma, squared=False)

plot_acf(model_fit_arma.resid)
plot_pacf(model_fit_arma.resid)
mean_squared_error(splitted_data['Test'],yhat_arma, squared = True) #MSE
mean_squared_error(splitted_data['Test'],yhat_arma, squared = False) #RMSE



splitted_data = split(data = data, diff_faktor= 0, rel_faktor = 0.9 )
model_fit_sari = SARIMAX(splitted_data['Train'], order = (5,0,3), seasonal_order=(7,0,1,7), enforce_stationarity=False, enforce_invertibility = False).fit(disp = 0)
yhat_sarimax = model_fit_sari.forecast(nr) 
mean_squared_error(splitted_data['Test'],yhat_sarimax, squared=False)

plot_acf(model_fit_sari.resid)
plot_pacf(model_fit_sari.resid)


data = ele

obs = splitted_data['Test'][0]
history.append(obs)
mod.forecast()

ele = y
ele.index = x
res = compare_models('persistance', ele)

#%%



   series = pd.Series(data.values, index=data.index)
    if len(dependent_var)*0.5 >= 50: 
        lags_length = 50
    else:
        lags_length = len(dependent_var)*0.5 - 5
    
    decomposition = seasonal_decompose(series, model = 'multiplicative', freq=12)

    decomposition.plot()
res_pers = res
res_arima = res
res_sari = res
res_sari_1 = res
res_pers [0]
res_arima [0]
res_sari [0] 
plt.plot(subresults)


plt.plot(da['Test'])
plt.plot(predictions_pers, color='red')


decomposition_stat = decompose_and_plot(y_stationary, x_stationary, rolling_window)  
# because peak is at 7 for both both pACF and ACF we would use AR and MA 
values = decomposition_stat.observed - decomposition_stat.seasonal
decompose_and_plot(decomposition_stat.resid, decomposition_stat.resid.index, rolling_window)  

prediction_nr = 30
X = decomposition_stat.resid.values
train, test = X[1:len(X)-prediction_nr], X[len(X)-prediction_nr:]




history = [x for x in train]
predictions = list()
for i in range(len(test)):
    model_fit = ARIMA(history, order=(4,7,3) ).fit(disp=0)
    # predict
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted={}, Expected={}'.format(yhat, obs))

mse = np.square(test - new).mean()
rmse = np.sqrt(mse)
print('RMSE: %.3f' % rmse)

plt.plot(test)
plt.plot(new , color = 'red')

len(predictions)
flattend = [y for x in np.asarray(predictions) for y in x]
new = flattend + decomposition_stat.seasonal[:30].values


hey.trend.dropna(inplace=True)

len(hey.trend)
z = pd.Series(test)

history.values
dtype(history)

decomposition_stat.trend.dropna(inplace =True)



history = [x for x in train]
predictions = list()
for i in range(len(test)):
    model_fit = ARMA(history, order=(6,3)).fit(disp=0)
    # predict
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted={}, Expected={}'.format(yhat, obs))

mse = np.square(test - predictions).mean()
rmse = np.sqrt(mse)
print('RMSE: %.3f' % rmse)




OLD: 10857.633 < 6853.547




model = ARIMA(da['Train'], order=(9, 1, 0) ).fit(disp=-1)
X[1:10-7][:,0:4]
X[1:10-7][:,4]
model
model_arima = ARIMA_and_plot( AR_value = 10, MA_value = 9 , orig_data = y,
              diff_faktor = diff_faktor, start = 0, end= 650, model = None)

ar1 = np.array([1, 0.33])
ma1 = np.array([1, 0.9])
simulated_ARMA_data = ArmaProcess(ar1, ma1).generate_sample(nsample=10)



ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
from statsmodels.tsa.arima_process import ArmaProcess
simulated_ARMA2_data = ArmaProcess(ar1, ma1).generate_sample(nsample=10000)
decomposition.trend + decomposition.seasonal + decomposition.resid



ar = np.arange(1,10,1)
ma = np.arange(1,10,1)

model_tuple = []
for ar_step in ar:
    for ma_step  in ma:
        model_tuple.append((ar_step, ma_step))


'''
Best AIC for (6,8) = 15951.776474396542 -> best model
'''
res = optimize_model(order_list, exog, use_ARMA = True)


try:
    order = res[0]
except:
    order = (6,8)
    
model = ARMA(y_diff, order=order).fit(disp=-1)
model.summary()
plt.plot(model.resid)
model.predict(params=y_diff[:3].values, start = 1 )
warnings.filterwarnings('default')


#%%##



to_nr = 800
x = data['Verkaufsdatum'][:to_nr ]
y = data['Einzel Menge in ST'][:to_nr ]


diction = split_dataset_for_training(x,y, 0.8)

history = [x for x in diction['y_train']]

model.summary()
predictions = list()
for t in tqdm(range(len(diction['y_test']))):
    model = ARMA(diction['y_train'], order=order).fit(disp=-1)
    output = model.forecast()
    yhat = output[0]
    predictions.append(yhat)
    
output = model.forecast(160)

plt.plot(output[0], color = 'blue', label = 'predicted')
plt.plot(diction['y_test'][:160].values, color = 'red', label = 'original')
plt.legend(loc = 'best')
warnings.filterwarnings('default')

#%%

#%%
def split_dataset_for_training(Timeseries_x, dependent_variable_y, rel_faktor):
    
    splited_dict = {'X_train':None, 'X_test':None, 'y_train':None, 'y_test':None}
    
    splited_dict['X_train'] = Timeseries_x[:int(Timeseries_x.shape[0]*rel_faktor)]
    splited_dict['X_test'] = Timeseries_x[int(Timeseries_x.shape[0]*rel_faktor):]
    splited_dict['y_train'] = dependent_variable_y[:int(Timeseries_x.shape[0]*rel_faktor)]
    splited_dict['y_test'] = dependent_variable_y[int(Timeseries_x.shape[0]*rel_faktor):]

    return splited_dict 




def optimize_model(order_list, exog, use_ARMA = True):
    """
        Return dataframe with parameters and corresponding AIC
        
        order_list - list with (p, d, q) tuples
        exog - the exogenous variable
    """
    
    results = []
    
    for order in tqdm(order_list):
        try:
            if use_ARMA:
                model = ARMA(exog, order=order).fit(disp=-1)
            else:
                model = ARIMA(exog, order=order).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        results.append([order, model.aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df



        

def ARMA_and_plot(AR_value, MA_value, orig_data, diff_faktor, start, end, model = None):
    
    if not model: 
        model = ARMA(orig_data, order=(AR_value, MA_value)).fit(disp=-1)
    
    cutout = orig_data[start:end]
    cutout.index -= diff_faktor
    fig, (ax,ax1) = plt.subplots(2,1)
    ax.set_ylabel('Original vs. Predicted')
    fig = model.plot_predict(start,end, ax = ax)
    fig = ax.plot(cutout, label = 'Original data')
    fig = ax.legend(loc='best')    
    
    ax1.set_ylabel('Only Prediction')
    fig = ax1.plot(model.forecast(end - len(model.resid))[0], label = 'forecast' )
    fig = ax1.legend(loc='best')    
    # fig.suptitle('Performance of ARMA')    
    return model
    


def ARIMA_and_plot(AR_value, MA_value, orig_data, diff_faktor, start, end, model = None):
    
    if not model: 
        model = ARIMA(orig_data, order=(AR_value, diff_faktor, MA_value) ).fit(disp=-1)
    
    cutout = orig_data[start:end]
    cutout.index -= diff_faktor
    fig, (ax,ax1) = plt.subplots(2,1)
    ax.set_ylabel('Original vs. Predicted')
    fig = model.plot_predict(start,end, ax = ax)
    fig = ax.plot(cutout, label = 'Original data')
    fig = ax.legend(loc='best')    
    
    ax1.set_ylabel('Only Prediction')
    fig = ax1.plot(model.forecast(end - len(model.resid))[0], label = 'forecast' )
    fig = ax1.legend(loc='best')    
    # fig.suptitle('Performance of ARMA')    
    return model
    



df = diction['y_train']
df = data['Einzel Menge in ST']
rolling_mean = df.rolling(window = 12).mean()
rolling_std = df.rolling(window = 12).std()
plt.plot(df, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()

result = adfuller(df['Passengers'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
    
df_log = np.log(df)
df_log = y_diff
plt.plot(df_log)



rolling_mean = df_log.rolling(window=12).mean()
df_log_minus_mean = df_log - rolling_mean
df_log_minus_mean.dropna(inplace=True)
get_stationarity(df_log_minus_mean, 12)

rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
df_log_exp_decay = df_log - rolling_mean_exp_decay
df_log_exp_decay.dropna(inplace=True)
get_stationarity(df_log_exp_decay, 12)

df_log_shift = df_log - df_log.shift()
df_log_shift.dropna(inplace=True)
get_stationarity(df_log_shift, 30)

series = pd.Series(data['Einzel Menge in ST'].values, index=data['Verkaufsdatum'])
decomposition = seasonal_decompose(series, model='additive')
# decomposition = seasonal_decompose(data) 
model = ARMA(df_log, order=(6,8))
results = model.fit(disp=-1)
plt.plot(df_log)
plt.plot(results.fittedvalues, color='red')

predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(df_log.iloc[0], index=df_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(df)
plt.plot(predictions_ARIMA)


# Given that we have data going for every month going back 12 years and want to forecast the number of passengers for the next 10 years, we use (12 x12)+ (12 x 10) = 264.

results.plot_predict(700,900)
