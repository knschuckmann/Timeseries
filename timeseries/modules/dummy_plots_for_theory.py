#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:12:24 2020

@author: Konstantin Schuckmann
"""


# Important imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import scipy.stats as stats
from timeseries.modules.config import dummy_temperature_path, dummy_flight_path, \
    dummy_save_path_section_2, dummy_save_path_section_3
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def save_fig(name, path_img, fig = None):
    if not os.path.exists(path_img):
        os.mkdir(path_img)
    else:
        if fig == None:
            fig = plt.gcf()
            
        fig.savefig(os.path.join(path_img, name + '.eps'), format='eps', bbox_inches='tight')          
    
def plot_random_data_for_stationarity_understanding(random_state, save_path, save=False):
    # Random plot for non stationarity and stationarity
    random_data_y = random_state.normal(4, 1, 7)
    x = np.arange(1, 8)
    
    fig, ax = plt.subplots() 
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(x, random_data_y)
    fig.show()
        
    if save:
        save_fig('random_data', save_path)

def autocorr_fkt(x, ax=None):
    # Same as plot_acf
    x = x.values.squeeze() # remove single-dimensional entries from the shape of an array
    if ax is None:
        ax = plt.gca() # get current axes 
    
    x = x - x.mean()
    autocorr = np.correlate(x, x, mode='full') #cross correlation but convolving 
    autocorr = autocorr[x.size:]
    autocorr /= autocorr.max()
    # stem plot plots vertical lines 
    return ax.stem(autocorr)

def temperature_data_preprocess_and_plot(data_path, save_path, save = False):
    # Preprocess temperature
    temp_data = pd.read_csv(data_path, parse_dates=['Date'])
    temp_data = temp_data.sort_values(by = 'Date', ignore_index = True)
    year_2015 = temp_data[temp_data.Date.dt.year == 2015 ]
    
    fig, ax = plt.subplots() 
    ax.plot(year_2015.Date.dt.strftime('%m-%d'),year_2015.Present_Tmax, label = 'South Korean temperature in 2015')
    ax.legend(frameon=False)
    ax.set_xticklabels(labels = year_2015.Date.dt.strftime('%m-%d').unique()[::3] , rotation = 45, fontsize=8)
    ax.set_xticks(ax.get_xticks()[::3])
    ax.set_xlabel('date')
    ax.set_ylabel('temperature in Celsius')
    fig.show()
    if save:
        save_fig('south_korea', save_path)
  
        
def plot_random_correlogram(random_state, save_path, save = False):
    # Correlogramm
    # np.random.seed(seed=0)
    # random_data = np.random.normal(0,2,1000)
    random_state.set_state(state)
    random_data = random_state.normal(0,2,1000)
    fig, axes = plt.subplots()
    fig = plot_acf(random_data, zero = True, lags = 40, title = 'Correlogram', ax = axes, markerfacecolor='black', color = 'black')
    axes.set_xlabel("Lag")
    axes.set_ylabel("ACF")
    fig.show()
    
    if save:
        save_fig('correlogram', save_path)
        
def flight_data_preprocess_and_plot_fundamentals(data_path, save_path, save_list = [False,False]):
    
    flight_data = pd.read_csv(data_path, parse_dates=['Month'])
    fig, ax = plt.subplots() 
    ax.plot(flight_data.Month,flight_data.Passengers, label = 'passengers')
    ax.legend(frameon=False)
    ax.set_xlabel('year')
    ax.set_ylabel('flight passenger volume')
    fig.show()
    
    if save_list[0]:
        save_fig('flight', save_path)
    
    flight_passengers = flight_data.Passengers
    flight_passengers.index = flight_data.Month
    fig, ax = plt.subplots()
    autocorr_fkt(flight_passengers[:])
    ax.set_xlabel('lag(month)')
    ax.set_ylabel('autocorrelation')
    fig.show()
    
    if save_list[1]:
        save_fig('ac_flight', save_path)
        

def flight_data_preprocess_and_plot_used_Models(data_path, save_path, save_list = [False,False,False]):
    flight_data = pd.read_csv(data_path, parse_dates=['Month'])

    for i in (1,10,4):
        rolling_mean = flight_data.iloc[:,1].rolling(window = i).mean()
        fig, ax = plt.subplots() 
        if i != 4:    # Moving average
            ax.plot(flight_data.Month,flight_data.Passengers, label = 'passengers')
            ax.plot(flight_data.Month,rolling_mean, label = 'ma_passengers')
            ax.legend(frameon=False)
            ax.set_xlabel('year')
            ax.set_ylabel('flight passenger volume')
            if save_list[0]:
                save_fig('ma_flight_' + str(i), save_path)
        else:    # Autocorrelation
            error_rate = flight_data.Passengers - rolling_mean
            fig = plot_acf(error_rate[i:], zero = True, lags = 50, title = 'Correlogram Passenger data MA(' + str(i-1) + ')' , ax = ax, markerfacecolor='black', color = 'black')
            ax.set_xlabel("Lag")
            ax.set_ylabel("ACF")
            if save_list[1]:
                save_fig('ma_flight_acf', save_path)
        fig.show()
        
    # PACF correlgram on passenger Data
    fig, ax = plt.subplots()
    fig = plot_pacf(flight_data.Passengers, zero = True, lags = 50, method = 'OLS', title = 'Partial correlogram on Passenger data' , ax = ax, markerfacecolor='black', color = 'black')
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    fig.show()
    if save_list[2]:
        save_fig('ma_flight_pacf', save_path)
    

def plot_random_data_correlogram_and_autocorr(random_state, state, save_path, save_list = [False,False]):
    # MA correlogram simulated data
    theta = pd.DataFrame([0.8,0.6,0.4])
           
    white_noise = pd.DataFrame(random_state.normal(0,4,1000))
    error_rate = pd.DataFrame(random_state.normal(10,10,1000))
    for t in range(4,1000):
        for nr, thet in enumerate(theta):
            error_rate[0][t] = thet * white_noise[0][t-nr]
        
    fig, axes = plt.subplots()
    fig = plot_acf(error_rate, zero = True, lags = 50, title = 'Correlogram simulated data MA(' + str(4-1) + ')' , ax = axes, markerfacecolor='black', color = 'black')
    axes.set_xlabel("Lag")
    axes.set_ylabel("ACF")
    
    if save_list[0]:
        save_fig('ma_simulate_acf', save_path)

    
    # AR model 
    random_state.set_state(state)
    random_data = random_state.normal(5,8,100)
    white_noise = random_state.normal(0,4,100)
    
    for nr in range(1,len(random_data)):
        random_data[nr] = 0.7 * random_data[nr-1] + white_noise[nr]
    
    
    ax = plt.subplot(212)   
    ax.text(.5,.9,'simulated timeseries data for AR(1)', horizontalalignment = 'center',transform=ax.transAxes) 
    ax = plt.plot(range(len(random_data)),random_data)
    
    ax1 = plt.subplot(221)
    plot_acf(random_data, ax = ax1, markerfacecolor='black', color = 'black')
    ax2 = plt.subplot(222)
    plot_pacf(random_data, zero = False, ax = ax2, markerfacecolor='black', color = 'black')
    if save_list[1]:
        plt.savefig(os.path.join(save_path,'ar_simulate.eps'), format='eps', bbox_inches='tight')
        
def plot_sigmoid_and_step_fkt(save_path, save = False):
    x_step = np.arange(-4,5,1)
    y_step = sum([list(np.repeat(0,5)), list(np.repeat(1,4))],[])
    x_sig = np.linspace(-4,4,100)
    y_sig =  1/(1 + np.exp(-x_sig)) 
    
    for name in ('step', 'sigmoid'):
        fig, ax = plt.subplots()
        if name == 'step':
            ax.plot(x_step,y_step, drawstyle='steps-pre')
        else:
            ax.plot(x_sig,y_sig)
        
        ax.set_title(name + ' function')
        ax.set_xlabel('Z')
        ax.set        
        if save:
            save_fig(name + '_function', save_path)

def set_working_directory():
    # if dir not in modules change to that folder
    if os.path.basename(os.getcwd()) != 'modules':
        os.chdir(os.path.join(os.path.dirname(os.getcwd())
                              ,'Timeseries/timeseries/modules'))

if __name__ == '__main__':    
    set_working_directory()
    
    # Set random State
    random_state = np.random.RandomState(0)
    state = random_state.get_state()

    ##################################
    ### Section Fundamentals plots ###
    ##################################    
    random_state.set_state(state)
    plot_random_data_for_stationarity_understanding(random_state = random_state, save_path = dummy_save_path_section_2, 
                                                    save = False)
    
    # Temperature data
    temperature_data_preprocess_and_plot(data_path = dummy_temperature_path,
                                         save_path = dummy_save_path_section_2,
                                         save = False)    
    # Flight data
    flight_data_preprocess_and_plot_fundamentals(data_path = dummy_flight_path,
                                                 save_path = dummy_save_path_section_2,
                                                 save_list = [False,False])
    
    ##################################    
    ### Section Used Models plots ###
    ##################################    
    random_state.set_state(state)
    plot_random_correlogram(random_state = random_state, save_path = dummy_save_path_section_3, 
                            save = False) 
    
    flight_data_preprocess_and_plot_used_Models(data_path = dummy_flight_path,
                                                save_path = dummy_save_path_section_3,
                                                save_list = [False,False,False])
    
    random_state.set_state(state)
    plot_random_data_correlogram_and_autocorr(random_state = random_state, 
                                              state = state, 
                                              save_path = dummy_save_path_section_3,
                                              save_list = [False,False])
    
    plot_sigmoid_and_step_fkt(save_path = dummy_save_path_section_3,
                              save = False)


    
    
    
    