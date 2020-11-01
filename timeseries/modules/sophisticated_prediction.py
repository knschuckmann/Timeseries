#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:29:32 2020

@author: Konstantin Schuckmann
"""
# Resource https://www.tensorflow.org/tutorials/structured_data/time_series
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf   
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error


from timeseries.modules.config import ORIG_DATA_PATH, SAVE_PLOTS_PATH, SAVE_MODELS_PATH, DATA, SAVE_RESULTS_PATH
from timeseries.modules.dummy_plots_for_theory import save_fig, set_working_directory
from timeseries.modules.load_transform_data import load_transform_excel
from timeseries.modules.baseline_prediction import combine_dataframe, monthly_aggregate, split

def disable_gpu(disable_gpu):
    if disable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # use CPU
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' # use GPU number 0 
        
     
def remove_unimportant_columns(all_columns, column_list):
    
    result_columns = set(all_columns)
    for column in column_list:
        try:
            result_columns -= set([column])
        except:
            continue
    return result_columns

def scale_data_to_standrad(train, test, val):
    # fit scaler
    scaler = StandardScaler()
    scaler = scaler.fit(train)
    # transform train
    train_set = scaler.transform(train)
    test_set = scaler.transform(test)
    val_set = scaler.transform(val)
    
    return scaler, train_set, test_set, val_set
    
    
def train_val_test(data):
    length = len(data)
    size_train = int(length*.7)
    size_test = int(length*.9)    
    # train, val, test
    return data[:size_train], data[size_train:size_test], data[size_test:]
    


class WindowGenerator():
    
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
      
        self.total_window_size = input_width + shift
      
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
      
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        'convert a lisr of consecutive inputs into a window of inputs and a window of labels'
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
      
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    def plot(self, model=None, plot_col='Einzel Menge in ST', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)
        
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
        
            if label_col_index is None:
                continue
        
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:
                predictions = model.predict(inputs)
                if len(predictions.shape) == 2:
                    pred = [predictions[i:i+self.label_width] for i in range(len(predictions))]
                    predictions = tf.convert_to_tensor(pred[:-self.label_width - 1])

                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)
            if n == 0:
                plt.legend()
      
        plt.xlabel('Time [h]')
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=32,)
      
        ds = ds.map(self.split_window)
      
        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result



class LSTM_Initial(tf.keras.Model):
    
    def __init__(self, batch_size, max_epoch_size, window_class_mod):
        super(LSTM_Initial, self).__init__()
        
        self.max_epoch_size = max_epoch_size
        self.batch_size = batch_size
        self.window_class_mod = window_class_mod
    
        self.LSTM1 = tf.keras.layers.LSTM(batch_size,return_sequences=False)#, 
                                          # input_shape=(self.batch_size,))
        # self.Dense = tf.keras.layers.Dense(units=1)
        self.Dense = tf.keras.layers.Dense(units=self.window_class_mod.label_width * len(self.window_class_mod.label_columns),
                                           kernel_initializer=tf.initializers.zeros)
        
    def call(self, inputs):
        x = self.LSTM1(inputs)
        return self.Dense(x)
         
    def train(self, patience=2):
        
        logdir = "../docs/log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")        
        # tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            update_freq='epoch')
        
        self.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
        return self.fit(self.window_class_mod.train, epochs=self.max_epoch_size,
                        validation_data=self.window_class_mod.val,callbacks=[early_stopping])
                        # callbacks=[early_stopping, tensorboard_callback])
    
    def who_am_i(self):
        return type(self).__name__
    
    def build_graph(self):
         x = tf.keras.Input(shape=(28,5))#,self.window_class_mod.example[0].shape[1:], ))
         return tf.keras.Model(inputs=x, outputs=self.call(x))

class LSTM_one(tf.keras.Model):
    
    def __init__(self, batch_size, max_epoch_size, window_class_mod):
        super(LSTM_one, self).__init__()
        
        self.max_epoch_size = max_epoch_size
        self.batch_size = batch_size
        self.window_class_mod = window_class_mod
        
        self.LSTM1 = tf.keras.layers.LSTM(batch_size,return_sequences=True)
        self.LSTM2 = tf.keras.layers.LSTM(batch_size,return_sequences=False)
        # self.Dense = tf.keras.layers.Dense(units=1)
        self.Dense = tf.keras.layers.Dense(units=self.window_class_mod.label_width * len(self.window_class_mod.label_columns),
                                           kernel_initializer=tf.initializers.zeros)
    
    def call(self, inputs):
        x = self.LSTM1(inputs)
        x = self.LSTM2(x)
        return self.Dense(x)
    
    def train(self, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
        self.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
        return self.fit(self.window_class_mod.train, epochs=self.max_epoch_size,
                        validation_data=self.window_class_mod.val,
                        callbacks=[early_stopping])
         
#%%
if __name__ == '__main__':
    set_working_directory()
    data_frame = load_transform_excel(ORIG_DATA_PATH)
    # data_orig = data_frame[0]#[['Einzel Menge in ST', 'Verkaufsdatum']]
    val_performance = {}
    performance = {}
    
    try:
        ran
    except:
        combined_m_df, combined_df = combine_dataframe(data_frame, monthly = True)
        monthly_list = list()
        for df in data_frame:
            monthly_list .append(monthly_aggregate(df, combined = False))
        ran = True        
        
    # set the DataFrame to predict 
    data_orig = combined_df
    
    used_columns = remove_unimportant_columns(data_orig.columns, ['Verkaufsdatum','Tages Wert in EUR','Einzel Wert in EUR','4Fahrt Wert in EUR', 'Gesamt Wert in EUR'])
    df = data_orig[used_columns]
    # num_features = df.shape[1]
    # column_indices = {name: i for i, name in enumerate(df.columns)}
    
    # Split data 
    train_df, val_df, test_df = train_val_test(df)
    
    # scale data
    scaler, sc_train, sc_test, sc_val = scale_data_to_standrad(train_df,test_df, val_df)
    sc_train_df = pd.DataFrame(sc_train, columns=train_df.columns, index = train_df.index)
    sc_test_df = pd.DataFrame(sc_test, columns=test_df.columns, index = test_df.index)
    sc_val_df = pd.DataFrame(sc_val, columns=val_df.columns, index = val_df.index)

    # initialize validation and performance parameters
   
    # compare = pd.DataFrame(columns=['Modelname','input_width', 'label_width', 'shift', 'RMSE', 'batch_size','val_performance'])
    input_w = pd.DataFrame([7,14,18,24], columns = ['input_width'])
    label_w = pd.DataFrame([1,2,3,4,5,6,7,14,18,24], columns = ['label_width'])
    shift = pd.DataFrame([1,2,3,4,5,6,7,14,18,24], columns = ['shift'])
    batch_size = pd.DataFrame([4,8,16,32,64,128], columns = ['batch_size'])

    for nr, width_inp in input_w.iterrows():
        for nr, width_label in label_w.iterrows():
            if int(width_label['label_width']) >= int(width_inp['input_width']) + int(shi_ft['shift']):
                continue
            for nr, shi_ft in shift.iterrows():
                
                for nr, batch in batch_size.iterrows():
                    # initialize Window for Timeseries analysis
                    # input width:= predict on behalf of the past length of input‚ width  
                    # label_width:= the amount of time points to predict
                    # shift:= inputwidth + shift position to start predicting from backwards. 
                    w2 = WindowGenerator(input_width= int(width_inp['input_width']) , label_width=int(width_label['label_width']),
                                         shift=int(shi_ft['shift']), train_df = sc_train_df, val_df = sc_val_df, 
                                         test_df = test_df, label_columns=['Einzel Menge in ST']) 
                    lstm_model = LSTM_Initial(batch_size = int(batch['batch_size']), max_epoch_size = 40, window_class_mod = w2)
                    # lstm_model = LSTM_one(batch_size = 32, max_epoch_size = 30, window_class_mod = w2)
                    name = lstm_model.who_am_i()
                    histroy = lstm_model.train()
            
                    # w2.plot(lstm_model)
                    val_performance[name] = lstm_model.evaluate(w2.val)
                    performance['MSE'] = lstm_model.evaluate(w2.test, verbose=1)
                    performance['RMSE'] = np.sqrt(performance['MSE'][0])
        
                    temp = pd.Series({'Modelname':name, 'input_width':int(width_inp['input_width']),
                                      'label_width':int(width_label['label_width']), 'shift':int(shi_ft['shift']),
                                      'batch_size':int(batch['batch_size']),'RMSE':performance['RMSE'], 
                                      'val_performance':val_performance[name]})
                    compare = compare.append(temp, ignore_index = True)
    compare.to_csv(SAVE_RESULTS_PATH + DATA + '_' +name +'.csv', sep=';')
    
#%%
   
    
res_1 = compare

res_1.to_csv(SAVE_RESULTS_PATH + DATA + '_' +name +'_3.csv')
w2 = WindowGenerator(input_width=7, label_width=3, shift=1, train_df = sc_train_df, val_df = sc_val_df, 
                      test_df = test_df, label_columns=['Einzel Menge in ST']) 

lstm_model = LSTM_Initial(batch_size = 32, max_epoch_size = 30, window_class_mod = w2)
histroy = lstm_model.train()

# w2.plot(lstm_model)
val_performance['LSTM_Initial'] = lstm_model.evaluate(w2.val)
performance['LSTM_Initial'] = lstm_model.evaluate(w2.test, verbose=1)

w2.plot(lstm_model)

*a,b = 1,2,3
b,c = a[0:2]

inputs, labels = w2.example

inputs.shape
labels.shape
predictions = lstm_model.predict(inputs)
predictions.shape
#important
pred = [predictions[i:i+w2.label_width] for i in range(len(predictions))]
predictions = tf.convert_to_tensor(pred[:-2])

p.shape
pred[:][:][0:3]

temp = pd.DataFrame(pred)

np.array(temp)
predictions = tf.convert_to_tensor(temp)


predictions = tf.convert_to_tensor(predictions)
predictions.set_shape([None,w2.label_width,None])
        labels.set_shape([None, self.label_width, None])
labels[0, :, 0]
predictions[]

pred = tf.expand_dims(predictions, axis = 2)

k = [predictions[i:i+3] for i in range(len(predictions))]

pd.DataFrame(k)


pred.set_shape([None,3,None])

n= 0

plt.plot(w2.input_indices, inputs[n, :, 0],
                     label='Inputs', marker='.', zorder=-10)
plt.scatter(range(13), labels[n:13, :, 0],
            edgecolors='k', label='Labels', c='#2ca02c', s=64)
plt.scatter(range(13), predictions[:13] ,
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)


#%%
#%%


labels[1, :, 0]
predictions
len(predictions.shape)
pred = scaler.inverse_transform(predictions)
orig = scaler.inverse_transform(labels[:,0,0])
labels.shape
plt.plot(pred, label = 'pred')
plt.plot(orig, label = 'orig')
plt.legend(loc = 'best')

mse,score = lstm_model.evaluate(w2.test, verbose=1, )
np.sqrt(mse)
np.sqrt(mean_squared_error(orig, pred))

labels
[0, :, 0]
predictions[0, :, 0]

#%%
class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index
    
    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


w2 = WindowGenerator(input_width=560, label_width=81, shift=1, train_df = sc_train_df, sc_val_df = sc_val_df, 
                      test_df = test_df,label_columns=['Einzel Menge in ST']) 
    
baseline = Baseline(label_index=column_indices['Einzel Menge in ST'])


baseline.compile(loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    
val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(w2.val)
performance['Baseline'] = baseline.evaluate(w2.test, verbose=1)


w2.plot(baseline)

#%%


    
    

    
    
