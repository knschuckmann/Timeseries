#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:57:33 2020

@author: Konstantin Schuckmann
"""


# A problem with ARIMA is that it does not support seasonal data. That is a time series with a repeating cycle.
# ARIMA expects data that is either not seasonal or has the seasonal component removed, e.g. seasonally adjusted via methods such as seasonal differencing.

from timeseries.modules.config import ORIG_DATA_PATH, SAVE_PLOTS_PATH, SAVE_MODELS_PATH, DATA
from timeseries.modules.dummy_plots_for_theory import save_fig, set_working_directory
from timeseries.modules.load_transform_data import load_transform_excel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import time

from statsmodels.tsa.stattools import adfuller # dickey fuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
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

def get_stationarity(timeseries, given_model, window, save_plots, print_results = False):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=window).mean()
    rolling_std = timeseries.rolling(window=window).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation for windowsize ' + str(window) )
    if save_plots:
        save_fig(name = given_model + '_mean_deviation_window_' + str(window), path_img=SAVE_PLOTS_PATH)
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

def plot_acf_pacf(series, given_model, save_plots, rolling_window):
    if len(series.values)*0.5 >= 60: # big value for seasonal AR and MA detection
        lags_length = 60
    else:
        lags_length = len(series.values)*0.5 - 5 # -5 because otherwise lags_length to long to display
        
    fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
    fig = plot_pacf(series, zero = True, lags = lags_length, method = 'OLS', title ='Partial correlogram to find out AR value for '+ given_model +' window ' + str(rolling_window), ax = ax1, markerfacecolor='black', color = 'black')
    fig = plot_acf(series, zero = True, lags = lags_length, title ='Correlogram to find out MA value for ' + given_model +' window ' + str(rolling_window), ax = ax2, markerfacecolor='black', color = 'black')
    plt.show(block= False)
    if save_plots:
        save_fig(name = given_model + '_acf_pacf_' + str(rolling_window), path_img=SAVE_PLOTS_PATH, fig = fig)
    
def decompose_and_plot(dependent_var, given_model, index, rolling_window, save_dec, save_acf, save_stat, print_results = False):
    
    series = pd.Series(dependent_var.values, index=index)
    decomposition = seasonal_decompose(series)
    decomposition.resid.dropna(inplace=True)
    figure = decomposition.plot()
    if save_dec:
        save_fig(name = 'decompose_window_' + str(rolling_window), path_img=SAVE_PLOTS_PATH, fig = figure)

    plot_acf_pacf(series, given_model, save_acf, rolling_window)
    
    plt.figure(2)
    dickey_fuller_results = get_stationarity(decomposition.observed, given_model = given_model, window = rolling_window, save_plots = save_stat, print_results = print_results)

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

def compare_models(given_model, data, diff_faktor, rolling_window, forecast_one_step = False):
    
    if forecast_one_step:
        name_supl = '_one_step'
    else:
        name_supl = ''
    
    print('\n', given_model, ' Model started:')
    subresults = pd.DataFrame(columns = ['Predicted', 'Expected'])
    result = {'Used Model':None, 'Model':None, 'MSE':None, 'RMSE':None, 'Orders':None}  
    decomposition = decompose_and_plot(dependent_var=data, given_model = given_model + name_supl, index=data.index, rolling_window=rolling_window, save_dec = True, save_acf=True, save_stat=True, print_results = True)
    if given_model in ['persistance','SARIMAX']:
        splitted_data = split(data = decomposition.observed, diff_faktor= 0, rel_faktor = 0.9 )
        if given_model == 'SARIMAX':
            order_dict = input_ar_ma(given_model)
        else:
            order_dict = None
            
    elif given_model in ['ARIMA', 'ARMA']: 
        diff_df = decomposition.observed.diff(diff_faktor)
        diff_df.dropna(inplace=True)
        print('\nAfter differencing:')
        get_stationarity(diff_df, given_model = given_model + name_supl, window= rolling_window, save_plots=True, print_results = True) # proove data is now stationary
        plot_acf_pacf(diff_df,given_model = given_model + name_supl, save_plots=True, rolling_window=diff_faktor)        
        splitted_data = split(data = diff_df, diff_faktor= 0, rel_faktor = 0.9 )
        order_dict = input_ar_ma(given_model)
    
    result['Orders'] = order_dict
    history = [x for x in splitted_data['Train']]    
    predictions = list()
    
    if forecast_one_step:
        test_length = 1
        splited_length = len(splitted_data['Test'])
    else:
        test_length = len(splitted_data['Test'])
        splited_length = 1
        
    for i in tqdm(range(splited_length)):
        # predict
        warnings.filterwarnings('ignore')
        if given_model == 'persistance':
            model_fit = None
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
        
        # print('>Predicted={}, Expected={}'.format(yhat, obs))
    result['Model'] = model_fit
    result['Used Model'] = given_model
    if given_model == 'persistance':   
        predictions = sum(predictions, [])
    else:
        predictions = predictions[0]
    result['MSE'] = np.round(mean_squared_error(splitted_data['Test'][:,0], predictions, squared = True),2)
    result['RMSE'] = np.round(mean_squared_error(splitted_data['Test'][:,0], predictions, squared = False),2)
    print('RMSE: %.3f' % result['RMSE'])
    warnings.filterwarnings('default')

    return [result, subresults]

def monthly_aggregate(data_frame, combined):
    if not combined:
        data_frame.index = data_frame['Verkaufsdatum']
        data_frame = data_frame.loc[:,data_frame.columns != 'Verkaufsdatum']
        
    result_df = pd.DataFrame(index = set(data_frame.index.to_period('M')), columns = data_frame.columns)
    
    for year_month in tqdm(set(result_df.index)):
        result_df.loc[year_month] = data_frame.loc[data_frame.index.to_period('M') == year_month].sum()
    
    return result_df.sort_index()


def combine_dataframe(data_frame_with_all_data, monthly= False, output_print = False):
    head_names = ['Einzel Menge in ST', '4Fahrt Menge in ST', 'Tages Menge in ST', 'Gesamt Menge in ST']
    df1 = data_frame_with_all_data[0]
    df1.index = df1['Verkaufsdatum']
    result_df = pd.DataFrame(columns = head_names)
    result_df[list(set(head_names) - set(['Tages Menge in ST']))] = df1[list(set(head_names) - set(['Tages Menge in ST']))]
    result_df['Tages Menge in ST'] = np.zeros(result_df.shape[0], dtype = int)
    
    for df in tqdm(data_frame[1:]):
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
                if output_print:
                    print('This header is not present in temp "{}" \n'.format(name))
                pass
    
    # insert new column
    result_df['Gesamt Menge in ST calc'] = result_df[['Einzel Menge in ST', '4Fahrt Menge in ST', 'Tages Menge in ST']].sum(axis = 1)
    
    if monthly:
        print('\nMonthly data aggregation:')
        time.sleep(0.3)
        monthly_df = monthly_aggregate(result_df, combined = True)
        return monthly_df, result_df
    
    return result_df    

def print_best_result(result_list):
    if type(result_list ) == dict:
        print('\nBest model {} with RMSE: {}'.format(result_list['Used Model'], result_list['RMSE'])) 
        return result_list
    else:
        for nr, res in enumerate(result_list):
            if nr == 0:
                min_temp = res['RMSE']
                temp_nr = nr
            elif res['RMSE'] < min_temp:
                temp_nr = nr
                min_temp = res['RMSE']
        print('\nBest model {} with RMSE: {}'.format(result_list[temp_nr]['Used Model'], result_list[temp_nr]['RMSE']))  
        return result_list[temp_nr]


if __name__ == '__main__':
    
    set_working_directory()
    data_frame = load_transform_excel(ORIG_DATA_PATH)
    
    combined_m_df, combined_df = combine_dataframe(data_frame, monthly = True)
    
    monthly_list = list()
    for df in data_frame:
        monthly_list .append(monthly_aggregate(df, combined = False))
        
    one_step = False
    data_to_use = data_frame[9]
    used_column = combined_df.columns[3]
    models = ['persistance', 'ARMA', 'ARIMA', 'SARIMAX']
    rolling_window = 7
    diff_faktor = 7
    
    result_list = list()
    for model in models:
        temp, pred = compare_models(given_model = model , data = data_to_use[used_column], 
                                diff_faktor = diff_faktor, rolling_window = rolling_window, forecast_one_step = one_step)
        result_list.append(temp)
        if model != 'persistance':
            temp['Model'].save(SAVE_MODELS_PATH + model + '_' + DATA +'.pkl')
    
    best_model = print_best_result(result_list)  
    result = pd.DataFrame(result_list)
    result.loc[result.shape[0]] = best_model
    result.to_csv(SAVE_MODELS_PATH + 'results.csv')


    # train_size = int(data_to_use.shape[0]*0.9)
    
    # if best_model['Used Model'] == 'ARMA':
    #     best_model['Model'].plot_predict(train_size-30, train_size+20)





