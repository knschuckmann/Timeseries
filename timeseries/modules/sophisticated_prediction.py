#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:29:32 2020

@author: Konstantin Schuckmann
"""
# Resource https://www.tensorflow.org/tutorials/structured_data/time_series
import os
import datetime
# import IPython
# import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf   
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from timeseries.modules.config import ORIG_DATA_PATH, SAVE_PLOTS_PATH, SAVE_MODELS_PATH, SAVE_RESULTS_PATH, ORIGINAL_PATH, MONTH_DATA_PATH,MODELS_PATH
from timeseries.modules.dummy_plots_for_theory import save_fig, set_working_directory
from timeseries.modules.load_transform_data import load_transform_excel
from timeseries.modules.baseline_prediction import combine_dataframe, monthly_aggregate
import warnings

def disable_gpu(disable_gpu):
    if disable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # use CPU
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' # use GPU number 0 

def cerate_lstm(batch_size_list=[64, 64], dropout=0.2, multisteps = 1, **kwargs):
    
    model = tf.keras.Sequential()
    
    batch_size_list = [3]
    for number in range(len(batch_size_list) -1):
        model.add(tf.keras.layers.LSTM(batch_size_list[number], return_sequences=True, **kwargs))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
    
    model.add(tf.keras.layers.LSTM(batch_size_list[-1], return_sequences=False, **kwargs))
    if dropout > 0:
        model.add(tf.keras.layers.Dropout(dropout))
    
    model.add(tf.keras.layers.Dense(units = multisteps))
    
    return model

def compile_and_fit(model, epochs, train , val, patience = 3, **kwargs):        
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    return model.fit(train, epochs=epochs, validation_data=val,
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
 

def train_val_test_split(data, rel_train, rel_val):
    length = len(data)
    size_train = int(length*rel_train)
    size_test = int(length*rel_val)    
    # train, val, test
    return data[:size_train], data[size_train:size_test], data[size_test:]
    
def scale_data_to_scaler(scale_method, train, test, val):
    # fit scaler
    scaler = scale_method
    scaler = scaler.fit(train)
    # transform train
    train_set = scaler.transform(train)
    test_set = scaler.transform(test)
    val_set = scaler.transform(val)
    return scaler, train_set, test_set, val_set

def check_if_column_right_place(column, dataframe):
    
    index_name = dataframe.index.name
    first_series  = dataframe[dataframe.columns[np.where(dataframe.columns == column)]]
    other_series = dataframe[dataframe.columns[np.where(dataframe.columns != column)]]
    return first_series.merge(other_series, left_on=index_name , right_on=index_name )

def reshape_array(numpy_array, timesteps):
    
    new_list = [numpy_array[x:x+timesteps] for x in range(int(np.ceil(numpy_array.shape[0]/timesteps)))]
    return np.array(new_list)

def split_into_in_out(train, val, test, prediction_steps):
    train_X, train_y = train[:, :-prediction_steps], train[:, -prediction_steps:]
    val_X, val_y = val[:, :-prediction_steps], val[:, -prediction_steps:]
    test_X, test_y = test[:, :-prediction_steps], test[:, -prediction_steps:]
    return train_X, train_y, val_X, val_y, test_X, test_y

def create_timeseries_data_batches(X, y, prediction_steps, batch_size):
    '''
    sequence_length: Length of the output sequences (in number of timesteps).
    batch_size: Number of timeseries samples in each batch (except maybe the last one).
    '''
    return tf.keras.preprocessing.timeseries_dataset_from_array(data = X, targets = y, 
                                                           sequence_length = prediction_steps, batch_size=batch_size)

def plot_losses(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    

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
    
        final_dict[monthly_name].index.name = 'date'
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
        final_dict['combined'].index.name = 'date'
    return final_dict 


    
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
 
def create_multivariate_data(data_orig, take_column, events, monthly=False, multivariate=True):
    # set the DataFrame to predict 
    
    data_orig_df = pd.DataFrame(data_orig[take_column], columns = [take_column])
    if multivariate:
        if monthly:
            data_orig_df = data_orig_df.merge(events, left_index = True, right_index=True)
            data_orig_df.index.name = events.index.name          
        else:
            data_orig_df = data_orig_df.merge(events, left_on =data_orig_df.index, right_on='date')
            data_orig_df = data_orig_df.set_index(data_orig_df['date'], drop = True, verify_integrity= True)
            data_orig_df = data_orig_df.drop('date', axis = 1)            

    df = check_if_column_right_place(column = take_column, dataframe = data_orig_df) 
    return df

    
def find_best_values(data_list, data_names_list, events, monthly, multivariate, 
                              find_parameters, parameters,plot_loss_and_results= False):
    multi = 'multiVar' if multivariate else 'uniVar'
    
    compare = pd.DataFrame(columns=['training_data','column_taken','used_timesteps_for_pred', 'prediction_steps', 'RMSE', 
                                  'batch_size_window','lstm_batch_size_list', 'multivariate'])

    # set the DataFrame to predict 
    # better to give only two datasets in list because training is taking very long
    for data_orig, data_name in zip(data_list, data_names_list):
        print(data_name)
        used_columns = remove_unimportant_columns(data_orig.columns, ['Verkaufsdatum',
                                                                      'Tages Wert in EUR',
                                                                      'Einzel Wert in EUR',
                                                                      '4Fahrt Wert in EUR', 
                                                                      'Gesamt Wert in EUR'])
        for column in used_columns:
            if 'Einzel' in column:
                take_column = column
        # print('dataframe: ',data_name, '\ncolumn: ',take_column)
        data_orig_df = pd.DataFrame(data_orig[take_column], columns = [take_column])

        df = create_multivariate_data(data_orig_df, take_column, events, monthly=monthly, 
                                      multivariate=multivariate)
        
        for used_timesteps_for_pred in find_parameters['used_timesteps_for_pred_list']:
            for batch_size_window in find_parameters['batch_size_window_list']:
                for lstm_batch_size_list in find_parameters['lstm_batch_size_list_list']:
                    
                    reframed = series_to_supervised(df, n_in = used_timesteps_for_pred, 
                                                    n_out = parameters['prediction_steps'])
                    
                    # Split data 
                    train_df, val_df, test_df = train_val_test_split(reframed, rel_train = .7, 
                                                                     rel_val = .9)
                    
                    # scale data
                    scaler, sc_train, sc_test, sc_val = scale_data_to_scaler(parameters['scale_method'], 
                                                                             train_df, test_df, 
                                                                             val_df)
                    
                    # split into input and outputs
                    train_X, train_y, val_X, val_y, test_X, test_y  = split_into_in_out(train = sc_train, val = sc_val, 
                                                                                        test = sc_test, 
                                                                                        prediction_steps = parameters['prediction_steps'])
                    # print(train_X.shape, train_y.shape, val_X.shape, val_y.shape,  test_X.shape, test_y.shape)
                    
                    # create Window 
                    data_window = {'Train':None, 'Val':None, 'Test':None}
                    for key, X, y in zip(data_window.keys(), [train_X, val_X, test_X],[train_y, val_y, test_y]):
                        data_window [key] = create_timeseries_data_batches(X= X, y= y, 
                                                                           prediction_steps= parameters['prediction_steps'], 
                                                                           batch_size = batch_size_window)
                    # Train model 
                    lstm_model = cerate_lstm(batch_size_list= lstm_batch_size_list, 
                                             dropout=.2, 
                                             multisteps = parameters['prediction_steps'])
                    history = compile_and_fit(lstm_model, parameters['epochs'], data_window['Train'],
                                              data_window['Val'], parameters['patience'], verbose = 0)
                    # lstm_model.save(os.path.join(MODELS_PATH, 'LSTM_monthly', data_name))                                        
                    if plot_loss_and_results: plot_losses(history)
        
                    # Predict
                    yhat = lstm_model.predict(data_window['Test'])
        
                    if parameters['prediction_steps'] >= 2:
                        test_X = test_X[:-(parameters['prediction_steps']-1)]
                        test_y = test_y[:-(parameters['prediction_steps']-1)]
                    
                    inv_yhat = np.concatenate((test_X, yhat), axis=1)
                    inv_yhat = scaler.inverse_transform(inv_yhat)
                    inv_yhat = inv_yhat[:,-parameters['prediction_steps']:]
                    
                    inv_y = np.concatenate((test_X, test_y), axis=1)
                    inv_y = scaler.inverse_transform(inv_y)
                    inv_y = inv_y[:,-parameters['prediction_steps']:]
                    
                    # Plot predictions
                    for i in range(parameters['prediction_steps']):
                        rmse = mean_squared_error(inv_y[:,i], inv_yhat[:,i], squared = False)
                        title = 'Plot of prediction for timestep (t)' if i == 0 else 'Plot of prediction for timestep (t+%d)' %(i)
                        if plot_loss_and_results:
                            plt.title(title)
                            plt.plot(inv_y[:,i], label = 'orig')
                            plt.plot(inv_yhat[:,i], label = 'pred')
                            plt.legend(loc = 'best')
                            plt.show()
                        print('Test RMSE: %.3f\n' % rmse)
        
                    temp = pd.Series({'training_data':data_name, 'column_taken':take_column, 
                                      'used_timesteps_for_pred':int(used_timesteps_for_pred),
                                      'prediction_steps':parameters['prediction_steps'], 
                                      'RMSE':np.round(rmse,2),
                                      'multivariate': str(multivariate),
                                      'batch_size_window':int(batch_size_window), 
                                      'lstm_batch_size_list':lstm_batch_size_list})
    
                    compare = compare.append(temp, ignore_index = True)
                                        
    compare.to_csv(os.path.join(SAVE_RESULTS_PATH, parameters['lstm_path']+'_find_params_'+
                                multi+'.csv'),
                   sep=';', decimal=',', index = False)

def create_pred_for_all_columns(data_list, data_names_list, 
                                            monthly, multivariate, events, 
                                            parameters, plot_loss_and_results= False):
    
    multi = 'multiVar' if multivariate else 'uniVar'
    compare = pd.DataFrame(columns=['training_data','column_taken','used_timesteps_for_pred', 'prediction_steps',
                                    'RMSE', 'batch_size_window','lstm_batch_size_list', 'multivariate'])
    # set the DataFrame to predict 
    data_orig,data_name = data_list[3], data_names_list[3]
    for data_orig, data_name in zip(data_list, data_names_list):
        used_columns = remove_unimportant_columns(data_orig.columns, ['Verkaufsdatum',
                                                                      'Tages Wert in EUR',
                                                                      'Einzel Wert in EUR',
                                                                      '4Fahrt Wert in EUR', 
                                                                      'Gesamt Wert in EUR'])
        for take_column in used_columns:
            print('dataframe: ',data_name, '\ncolumn: ',take_column)
            data_orig_df = pd.DataFrame(data_orig[take_column], columns = [take_column])

            df = create_multivariate_data(data_orig_df, take_column, events, monthly=monthly, 
                                          multivariate=multivariate)
                            
            reframed = series_to_supervised(df, n_in = parameters['used_timesteps_for_pred'], 
                                                                  n_out = parameters['prediction_steps'])
            
            # Split data 
            train_df, val_df, test_df = train_val_test_split(reframed, rel_train = .7, rel_val = .9)
            
            # scale data
            scaler, sc_train, sc_test, sc_val = scale_data_to_scaler(parameters['scale_method'], train_df, 
                                                                     test_df, val_df)
            
            # split into input and outputs
            train_X, train_y, val_X, val_y, test_X, test_y  = split_into_in_out(train = sc_train, val = sc_val, 
                                                                                test = sc_test, 
                                                                                prediction_steps = parameters['prediction_steps'])
            # print(train_X.shape, train_y.shape, val_X.shape, val_y.shape,  test_X.shape, test_y.shape)
            
            # create Window 
            data_window = {'Train':None, 'Val':None, 'Test':None}
            for key, X, y in zip(data_window.keys(), [train_X, val_X, test_X],[train_y, val_y, test_y]):
                data_window [key] = create_timeseries_data_batches(X= X, y= y, 
                                                                   prediction_steps= parameters['prediction_steps'], 
                                                                   batch_size = parameters['batch_size_window'])
            
            # Train model 
            lstm_model = cerate_lstm( batch_size_list= parameters['lstm_batch_size_list'], 
                                     dropout=.2, multisteps = parameters['prediction_steps'])
            history = compile_and_fit(lstm_model, parameters['epochs'], data_window['Train'],
                                      data_window['Val'], parameters['patience'], verbose = 0)
            
            lstm_model.save(os.path.join(MODELS_PATH, parameters['lstm_path'], multi, data_name, take_column))
            
            
            if plot_loss_and_results: plot_losses(history)
           
            # Predict
            yhat = lstm_model.predict(data_window['Test'])
            if prediction_steps >= 2:
                test_X = test_X[:-(parameters['prediction_steps']-1)]
                test_y = test_y[:-(parameters['prediction_steps']-1)]
            
            inv_yhat = np.concatenate((test_X, yhat), axis=1)
            inv_yhat = scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:,-parameters['prediction_steps']:]
            
            inv_y = np.concatenate((test_X, test_y), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
            inv_y = inv_y[:,-parameters['prediction_steps']:]
            
            # Plot predictions
            for i in range(parameters['prediction_steps']):
                rmse = mean_squared_error(inv_y[:,i], inv_yhat[:,i], squared = False)
                title = 'Plot of prediction for timestep (t)' if i == 0 else 'Plot of prediction for timestep (t+%d)' %(i)
                if plot_loss_and_results:
                    plt.title(title)
                    plt.plot(inv_y[:,i], label = 'orig')
                    plt.plot(inv_yhat[:,i], label = 'pred')
                    plt.legend(loc = 'best')
                    plt.show()
                print('Test RMSE: %.3f\n' % rmse)
                
                temp = pd.Series({'training_data':data_name, 'column_taken':take_column, 
                                  'used_timesteps_for_pred':parameters['used_timesteps_for_pred'],
                                  'prediction_steps':parameters['prediction_steps'], 
                                  'RMSE':np.round(rmse,2),
                                  'multivariate': str(multivariate),
                                  'batch_size_window':parameters['batch_size_window'], 
                                  'lstm_batch_size_list':parameters['lstm_batch_size_list']})

                compare = compare.append(temp, ignore_index = True)
    
    compare.to_csv(os.path.join(SAVE_RESULTS_PATH, parameters['lstm_path'] + '_performance_results_' + multi + '.csv'),
                   sep=';', decimal=',', index = False)


def create_pred_with_trained_model(path_to_model, data_list, data_names_list, 
                                            monthly, multivariate, name_trained_df, 
                                            name_trained_column, events, parameters, plot_loss_and_results= False):
    
    multi = 'multiVar' if multivariate else 'uniVar'
    compare = pd.DataFrame(columns=['Used Model','trained_df','Trained column', 'RMSE', 
                                    'Predicted column','Pred DataFrame'])  
    # set the DataFrame to predict 
    for data_orig, data_name in zip(data_list, data_names_list):
        used_columns = remove_unimportant_columns(data_orig.columns, ['Verkaufsdatum',
                                                                      'Tages Wert in EUR',
                                                                      'Einzel Wert in EUR',
                                                                      '4Fahrt Wert in EUR', 
                                                                      'Gesamt Wert in EUR'])
        for take_column in used_columns:
            print('dataframe: ',data_name, '\ncolumn: ',take_column)
            data_orig_df = pd.DataFrame(data_orig[take_column], columns = [take_column])

            df = create_multivariate_data(data_orig_df, take_column, events, monthly=monthly, multivariate=multivariate)
                            
            reframed = series_to_supervised(df, n_in = parameters['used_timesteps_for_pred'], n_out = parameters['prediction_steps'])
            
            # Split data 
            train_df, val_df, test_df = train_val_test_split(reframed, rel_train = .7, rel_val = .9)
            # scale data
            scaler, sc_train, sc_test, sc_val = scale_data_to_scaler(parameters['scale_method'], train_df, test_df, val_df)            
            # split into input and outputs
            train_X, train_y, val_X, val_y, test_X, test_y  = split_into_in_out(train = sc_train, val = sc_val, test = sc_test, prediction_steps = parameters['prediction_steps'])
            # print(train_X.shape, train_y.shape, val_X.shape, val_y.shape,  test_X.shape, test_y.shape)
            
            # create Window 
            data_window = {'Train':None, 'Val':None, 'Test':None}
            for key, X, y in zip(data_window.keys(), [train_X, val_X, test_X],[train_y, val_y, test_y]):
                data_window [key] = create_timeseries_data_batches(X= X, y= y, 
                                                                   prediction_steps= parameters['prediction_steps'], batch_size = parameters['batch_size_window'])
            
            # Load model
            lstm_model = tf.keras.models.load_model(path_to_model)
        
            # Predict
            yhat = lstm_model.predict(data_window['Test'])
            
            if parameters['prediction_steps'] >= 2:
                test_X = test_X[:-(parameters['prediction_steps']-1)]
                test_y = test_y[:-(parameters['prediction_steps']-1)]
            
            inv_yhat = np.concatenate((test_X, yhat), axis=1)
            inv_yhat = scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:,-parameters['prediction_steps']:]
            
            inv_y = np.concatenate((test_X, test_y), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
            inv_y = inv_y[:,-parameters['prediction_steps']:]
            
            # Plot predictions
            for i in range(parameters['prediction_steps']):
                rmse = mean_squared_error(inv_y[:,i], inv_yhat[:,i], squared = False)
                title = 'Plot of prediction for timestep (t)' if i == 0 else 'Plot of prediction for timestep (t+%d)' %(i)
                if plot_loss_and_results:
                    plt.title(title)
                    plt.plot(inv_y[:,i], label = 'orig')
                    plt.plot(inv_yhat[:,i], label = 'pred')
                    plt.legend(loc = 'best')
                    plt.show()
                print('Test RMSE: %.3f\n' % rmse)
                
                temp = pd.Series({'Used Model':'LSTM','trained_df':name_trained_df,
                                  'Trained column':name_trained_column, 'RMSE':np.round(rmse,2), 
                                  'Predicted column':take_column ,'Pred DataFrame':data_name})
                
                compare = compare.append(temp, ignore_index = True)
    
                                          
    compare.to_csv(os.path.join(SAVE_RESULTS_PATH, 'compare_'+name_trained_df+'_'+ parameters['lstm_path']+
                                '_trained_'+name_trained_column+'_'+multi + '.csv'), sep=';', decimal=',', index = False)
   
#%%
if __name__ == '__main__':
    # initialize
    set_working_directory()
    monthly_given_list = load_transform_excel(MONTH_DATA_PATH)
    monthly_names_given_list = ['aut', 'eigVkSt', 'privat', 'app']
    agg_monthly_names_list = ['einz_aut', 'einz_eigVkSt', 'einz_privat', 'einz_bus', 'einz_app',
                              'tages_aut', 'tages_eigVkSt', 'tages_privat', 'tages_bus', 'tages_app']
    try:
        ran
    except:
        data_frame = load_transform_excel(ORIG_DATA_PATH)
        events = pd.read_excel(ORIGINAL_PATH + 'events.xlsx')
        events_monthly = pd.read_excel(ORIGINAL_PATH + 'events_monthly.xlsx')
        _ , combined_df = combine_dataframe(data_frame, monthly = True)
        monthly_list = list()
        for df in data_frame:
            monthly_list .append(monthly_aggregate(df, combined = True))
        monthly_dict = create_dict_from_monthly(monthly_given_list, monthly_names_given_list, monthly_list, agg_monthly_names_list, True)    
        monthly_events = monthly_aggregate(events_monthly, False)
        data_frame = set_index_in_df_list(data_frame)
        ran = True 
    

    warnings.filterwarnings('ignore')
    used_timesteps_for_pred = 49
    prediction_steps = 1
    multivariate = False
    batch_size_window = 16
    epochs = 40
    patience = 3 # for early stoping
    lstm_batch_size_list = [50,50,50,50]
    scale_method = MinMaxScaler()
    plot_loss_and_results = False
    monthly = True
    event_df = events 
    multi = 'multiVar' if multivariate else 'uniVar'
    action = 'predict'       # 'find', 'train', 'predict'
    
    if monthly:
        lstm_path = 'LSTM_monthly'
        data_list = list(monthly_dict.values())
        data_names_list = list(monthly_dict.keys())
        used_timesteps_for_pred = 4
        used_timesteps_for_pred_list = [4, 7, 10, 30]
        event_df = monthly_events
        batch_size_window = 128
        
        if not multivariate:
            batch_size_window = 4
            lstm_batch_size_list = [20,20]
            column = 'Einzel'
    else:
        lstm_path = 'LSTM'
        data_list = data_frame[:]
        data_list.append(combined_df)
        data_names_list = ['df_0_einzel_aut', 'df_1_einzel_eigVkSt', 'df_2_einzel_privat',
                           'df_3_einzel_bus', 'df_4_einzel_app', 'df_5_tages_aut', 
                           'df_6_tages_eigVkSt', 'df_7_tages_privat', 'df_8_tages_bus', 
                           'df_9_tages_app', 'combined_df']
        column = 'Einzel Menge in ST'
        used_timesteps_for_pred_list = [7, 10, 30, 49, 50]
    find_parameters = {'used_timesteps_for_pred_list':used_timesteps_for_pred_list,'prediction_steps_list':[1],
                       'batch_size_window_list':[4,16,32,128],'lstm_batch_size_list_list':[20, 50, [20,20], [50,50], [50,50,50,50]]}
    parameters = {'used_timesteps_for_pred':used_timesteps_for_pred, 'prediction_steps':prediction_steps, 
              'scale_method':scale_method, 'batch_size_window':batch_size_window, 'lstm_batch_size_list':lstm_batch_size_list,
              'lstm_path':lstm_path, 'patience':patience,'epochs':epochs}
    
   
    if action == 'find':
        find_best_values(data_list = [data_list[len(data_list) - 1],data_list[0]], 
                                  data_names_list = [data_names_list[len(data_list) - 1],data_names_list[0]],
                                  events = event_df, monthly = monthly, 
                                  multivariate = multivariate, 
                                  find_parameters = find_parameters, parameters = parameters,
                                  plot_loss_and_results= False)
    elif action == 'train':
        create_pred_for_all_columns(data_list = data_list, data_names_list = data_names_list, 
                                              monthly = monthly , multivariate = multivariate,
                                              events = event_df, parameters = parameters, 
                                              plot_loss_and_results= plot_loss_and_results)
    else: 
        create_pred_with_trained_model(path_to_model = os.path.join(MODELS_PATH, lstm_path, 
                                                                              multi, 
                                                                              data_names_list[len(data_names_list) - 1], 
                                                                              column,''),
                                                data_list = data_list, data_names_list = data_names_list, 
                                                monthly = monthly,  multivariate = multivariate, 
                                                name_trained_df = 'combined_df', 
                                                name_trained_column = 'Einzel', 
                                                events = event_df,
                                                parameters = parameters, plot_loss_and_results= False)
       