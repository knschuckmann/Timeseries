
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:57:33 2020

@author: Konstantin Schuckmann
"""


# A problem with ARIMA is that it does not support seasonal data. That is a time series with a repeating cycle.
# ARIMA expects data that is either not seasonal or has the seasonal component removed, e.g. seasonally adjusted via methods such as seasonal differencing.

from timeseries.modules.config import ORIG_DATA_PATH, SAVE_PLOTS_PATH, SAVE_MODELS_PATH, \
    DATA, MONTH_DATA_PATH, MODELS_PATH, SAVE_RESULTS_PATH, SAVE_PLOTS_RESULTS_PATH_BASE
from timeseries.modules.dummy_plots_for_theory import save_fig, set_working_directory
from timeseries.modules.load_transform_data import load_transform_excel
# from timeseries.modules.sophisticated_prediction import create_dict_from_monthly
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import time
import glob
import os
import datetime

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
                                enforce_stationarity=True, enforce_invertibility = True).fit(disp = 0)
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
        if not forecast_one_step:
            predictions = predictions[0]
    result['MSE'] = np.round(mean_squared_error(splitted_data['Test'][:,0], predictions, squared = True),2)
    result['RMSE'] = np.round(mean_squared_error(splitted_data['Test'][:,0], predictions, squared = False),2)
    print('RMSE: %.3f' % result['RMSE'])
    warnings.filterwarnings('default')

    return [result, subresults]

def monthly_aggregate(data_frame, combined):
    if not combined:
        try:
            data_frame.index = data_frame['Verkaufsdatum']
            data_frame = data_frame.loc[:,data_frame.columns != 'Verkaufsdatum']
        except:
            data_frame.index = data_frame['date']
            data_frame = data_frame.loc[:,data_frame.columns != 'date']
    result_df = pd.DataFrame(index = set(data_frame.index.to_period('M')), columns = data_frame.columns)
    
    for year_month in tqdm(set(result_df.index)):
        result_df.loc[year_month] = data_frame.loc[data_frame.index.to_period('M') == year_month].sum()
    
    result_df.index.name = 'date'
    return result_df.sort_index()


def combine_dataframe(data_frame_with_all_data, monthly= False, output_print = False):
    head_names = ['Einzel Menge in ST', '4Fahrt Menge in ST', 'Tages Menge in ST', 'Gesamt Menge in ST']
    df1 = data_frame_with_all_data[0]
    df1.index = df1['Verkaufsdatum']
    result_df = pd.DataFrame(columns = head_names)
    result_df[list(set(head_names) - set(['Tages Menge in ST']))] = df1[list(set(head_names) - set(['Tages Menge in ST']))]
    result_df['Tages Menge in ST'] = np.zeros(result_df.shape[0], dtype = int)
    
    for df in tqdm(data_frame_with_all_data[1:]):
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


def create_dict_from_monthly(monthly_given_list, monthly_names_given_list, agg_monthly_list,
                             agg_monthly_names_list, combined = False):

    monthly_given_dict = {name:data for name, data in zip(monthly_names_given_list, monthly_given_list)}
    agg_monthly_dict = {name:data for name, data in zip(agg_monthly_names_list,agg_monthly_list)}
    
    monthly_dict_copy = {}
    for dic in tqdm(agg_monthly_dict):
        for dic1 in agg_monthly_dict:
            if dic != dic1 and dic.split('_')[1] == dic1.split('_')[1]:
                used_columns = remove_unimportant_columns(agg_monthly_dict[dic].columns, ['Verkaufsdatum','Tages Wert in EUR','Einzel Wert in EUR','4Fahrt Wert in EUR', 'Gesamt Wert in EUR'])
                used_columns1 = remove_unimportant_columns(agg_monthly_dict[dic1].columns, ['Verkaufsdatum','Tages Wert in EUR','Einzel Wert in EUR','4Fahrt Wert in EUR', 'Gesamt Wert in EUR'])
                temp = agg_monthly_dict[dic][used_columns].merge(agg_monthly_dict[dic1][used_columns1], left_index = True, right_index = True)
                temp['Gesamt Menge in ST'] = temp[['Gesamt Menge in ST_x','Gesamt Menge in ST_y']].sum(axis=1)
                monthly_dict_copy[dic.split('_')[1]] = temp.drop(['Gesamt Menge in ST_x','Gesamt Menge in ST_y'], axis = 1)
                
                lis = list()
                for nr,column in enumerate(monthly_dict_copy[dic.split('_')[1]].columns):
                    lis.append(column.split()[0])
                monthly_dict_copy[dic.split('_')[1]].columns = lis
    
    final_dict = {}
    for monthly_name, monthly_data in tqdm(monthly_given_dict.items()):
        einzel = monthly_data[(monthly_data['PGR'] == 200)]
        fahrt4 = einzel[einzel[einzel.columns[1]].str.contains('4-Fahrten|4 Fahrten', regex=True)]
        einzel = einzel[einzel[einzel.columns[1]].str.contains('4-Fahrten|4 Fahrten', regex=True) == False]
        tages = monthly_data[(monthly_data['PGR'] == 300)]
    
        final_df = pd.DataFrame([tages.sum(axis=0, numeric_only = True)[2:], 
                                 einzel.sum(axis=0, numeric_only = True)[2:],
                                 fahrt4.sum(axis=0, numeric_only = True)[2:]], 
                                index=['Tages', 'Einzel', '4Fahrt'])
        final_df = final_df.T

        las = list()
        for year_month in final_df.index:
            las.append(datetime.datetime.strptime(year_month, '%Y%m'))
        
        final_df.index = las
        final_df.index = final_df.index.to_period('M')
        final_df['Gesamt'] = final_df.sum(axis = 1)
        
        final_dict[monthly_name] = pd.concat([final_df, monthly_dict_copy[monthly_name].loc[ 
            pd.Period(max(final_df.index)+1):, : ]])
    
    if combined:
    
        tages = list()
        einzel = list()
        fahrt_4 = list() 
        gesamt = list()           
        final = pd.DataFrame()
        for key in final_dict.keys():
            tages.append( final_dict[key][final_dict[key].columns[0]])
            einzel.append( final_dict[key][final_dict[key].columns[1]])
            fahrt_4.append( final_dict[key][final_dict[key].columns[2]])
            gesamt.append( final_dict[key][final_dict[key].columns[3]])
        
        final['Tages'] = pd.DataFrame(tages).sum(axis = 0)
        final['Einzel'] = pd.DataFrame(einzel).sum(axis = 0)
        final['4Fahrt'] = pd.DataFrame(fahrt_4).sum(axis = 0)
        final['Gesamt'] = pd.DataFrame(gesamt).sum(axis = 0)
        
        final_dict['combined'] = final   
    return final_dict 

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


def remove_unimportant_columns(all_columns, column_list):
    
    result_columns = set(all_columns)
    for column in column_list:
        try:
            result_columns -= set([column])
        except:
            continue
    return result_columns

def predict_with_given_model(model, model_name, data, trained_column, used_column, data_name):

    print('\n', model_name, ' Model started:')
    if model_name in ['persistance','SARIMAX']:
        splitted_data = split(data = data, diff_faktor= 0, rel_faktor = 0.9 )
    elif model_name in ['ARIMA', 'ARMA']: 
        diff_df = decomposition.observed.diff(diff_faktor)
        diff_df.dropna(inplace=True)
        splitted_data = split(data = diff_df, diff_faktor= 0, rel_faktor = 0.9 )
    
    result = {'Used Model':model_name, 'Trained column':trained_column, 
              'RMSE':None, 'Predicted column':used_column, 'Pred DataFrame':data_name}  
    
    test_length = len(splitted_data['Test'])
    splited_length = 1
    
    for i in tqdm(range(splited_length)):
        # predict
        warnings.filterwarnings('ignore')
        yhat = model.predict(1,test_length)        
        
    obs = splitted_data['Test']
    res = np.concatenate((yhat.reshape(-1,1),obs), axis = 1)
    subresults = pd.DataFrame(res, columns = ['Predicted', 'Expected'])
    
    result['RMSE'] = np.round(mean_squared_error(obs, yhat, squared = False),2)
    print('RMSE: %.3f' % result['RMSE'])
    warnings.filterwarnings('default')

    final_result = pd.DataFrame.from_dict(result, orient = 'index').T
    
    return final_result, subresults

if __name__ == '__main__':
    
    set_working_directory()
    monthly_given_list = load_transform_excel(MONTH_DATA_PATH)
    monthly_names_given_list = ['aut', 'eigVkSt', 'privat', 'app']
    agg_monthly_names_list = ['einz_aut', 'einz_eigVkSt', 'einz_privat', 'einz_bus', 'einz_app',
                              'tages_aut', 'tages_eigVkSt', 'tages_privat', 'tages_bus', 'tages_app']
    try:
        ran
    except:
        data_frame = load_transform_excel(ORIG_DATA_PATH)
        combined_m_df, combined_df = combine_dataframe(data_frame, monthly = True)
        monthly_list = list()
        for df in data_frame:
            monthly_list .append(monthly_aggregate(df, combined = True))
        monthly_dict = create_dict_from_monthly(monthly_given_list, monthly_names_given_list, monthly_list, agg_monthly_names_list)    
        ran = True 
    
    one_step = False
    predict_pretrained = True
    # data_list = monthly_dict.values()
    # data_names_list = monthly_dict.keys()
    data_list = data_frame[:]
    data_list.append(combined_df)
    data_names_list = ['df_0_einzel_aut', 'df_1_einzel_eigVkSt', 'df_2_einzel_privat',
                        'df_3_einzel_bus', 'df_4_einzel_app', 'df_5_tages_aut', 
                        'df_6_tages_eigVkSt', 'df_7_tages_privat', 'df_8_tages_bus', 
                        'df_9_tages_app', 'combined_df']
    
    data_names_list  = data_names_list [10:]
    data_list = data_list[10:]
    data_to_use = combined_df
    used_column = combined_df.columns[0]
    # models = ['persistance', 'ARMA', 'ARIMA', 'SARIMAX']
    models = ['SARIMAX']
    rolling_window = 7
    diff_faktor = 7
    Path_to_models = os.path.join(MODELS_PATH, 'more_steps/Einzel/', DATA , '')
    
    # used_column = 'Einzel'
    if not predict_pretrained:
        result_list = list()
        for model in models:
            
            temp, pred = compare_models(given_model = model , data = data_to_use[used_column], 
                                    diff_faktor = diff_faktor, rolling_window = rolling_window, forecast_one_step = one_step)
            
            temp['name'] = data_name
            temp['used_column'] = used_column
            result_list.append(temp)
            # if model != 'persistance':
            #     temp['Model'].save(SAVE_MODELS_PATH + model + '_' + DATA +'.pkl')
        best_model = print_best_result(result_list)  

        plt.plot(pred[1], label = 'orig')                
        plt.plot(pred[0], label = 'pred')
        # plt.plot(pred[pred.columns[1]], label = pred.columns[1])
        # plt.plot(pred[pred.columns[0]], label = pred.columns[0])
        plt.title(model + ' Plot of predicted results with RMSE: ' + str(temp['RMSE']))
        plt.legend(loc='best')
        save_fig('sarimax_results_plot', SAVE_PLOTS_RESULTS_PATH_BASE)
        
        # result = pd.DataFrame(result_list)
        # result.loc[result.shape[0]] = best_model
        # result.to_csv(SAVE_MODELS_PATH + 'Baseline_results.csv')

        
            # train_size = int(data_to_use.shape[0]*0.9)
            
            # if best_model['Used Model'] == 'ARMA':
            #     best_model['Model'].plot_predict(train_size-30, train_size+20)
        
    else:
        final_res = pd.DataFrame(columns = ['Used Model', 'Trained column', 'RMSE', 'Predicted column', 'Pred DataFrame'])
        for path in glob.glob(Path_to_models + '*.pkl'):
            if os.path.basename(path).split('_')[0] in models:
                used_path = path
                model = models[0]
            else:
                print('No pkl file with given Model')
        
        if model in ['ARMA','ARIMA']:
            model_loaded = ARIMAResults.load(used_path)
        elif model == 'SARIMAX':
            model_loaded = SARIMAXResults.load(used_path)

        for data_to_use, data_name in zip(data_list, data_names_list):
            print(data_name)
            used_columns = remove_unimportant_columns(data_to_use.columns, 
                                                      ['Verkaufsdatum','Tages Wert in EUR','Einzel Wert in EUR','4Fahrt Wert in EUR', 'Gesamt Wert in EUR'])
            data_to_use = data_to_use[used_columns] 
            for used_column in data_to_use.columns:
               
                res, sub = predict_with_given_model(model_loaded, model, data_to_use[used_column], 
                                                    used_path.split('/')[-3], used_column.split()[0], data_name)
                
                final_res = final_res.append(res)
                plt.plot(sub[sub.columns[0]], label = sub.columns[0])
                plt.plot(sub[sub.columns[1]], label = sub.columns[1])
                plt.legend(loc='best')
                save_fig('sarimax_results_plot' + used_column, SAVE_PLOTS_RESULTS_PATH_BASE)
                plt.show()
        final_res.to_csv(os.path.join(SAVE_RESULTS_PATH, 'combined_df_' + model +'_trained_'+ 
                                      used_path.split('/')[-3] + '.csv'), 
                                      sep = ';', decimal = ',',  index = False)
        
        