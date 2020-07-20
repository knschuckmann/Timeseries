#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:12:24 2020

@author: Kostja
"""
#%%
# Global variables 
path_csv = '../resource_data/dummy/temperature/Bias_correction_ucl.csv'
path_img1 = '../../Latex/bhtThesis/Masters/2_fundamentals/pictures'
path_img2 = '../../Latex/bhtThesis/Masters/3_used_Models/pictures'


# Important imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

# save figure 
# Preprocess passengers for flight
flight_path_csv = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
flight_data = pd.read_csv(flight_path_csv, parse_dates=['Month'])

def save_fig(name, path_img):
    if not os.path.exists(path_img):
        os.mkdir(path_img)
    else:
        fig.savefig(path_img + '/' + name + '.eps', format='eps', bbox_inches='tight')  

#%%%

# Random plot for non stationarity and stationarity
np.random.seed(seed=0)
y = np.random.normal(4,1,7)
x = np.arange(1,8)

fig, ax = plt.subplots() 
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(x,y)
fig.show()
save_fig('non_stationarity')

#%%
# Plot of Gaus curve for teh stationarity plot 
mu = 0
sigma = 1
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
y = stats.norm.pdf(x, mu, sigma)

fig, ax = plt.subplots() 
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
#ax.axis('off')
ax.plot(x, y, color= 'k',linewidth=2)
fig.show()
save_fig('gaus_other_2')

#%%
# Preprocess temperature
temp_data = pd.read_csv(path_csv)
temp_data = temp_data.sort_values(by = 'Date')
temp_data = temp_data.reset_index(drop=True)

temp_data.Date = pd.to_datetime(temp_data.Date)

year_2015 = temp_data[temp_data.Date.dt.year == 2015 ]


#%%

fig, ax = plt.subplots() 
# fig.autofmt_xdate()
ax.plot(flight_data.Month,flight_data.Passengers, label = 'passengers')

ax.legend(frameon=False)
ax.set_xlabel('year')
ax.set_ylabel('flight passenger volume')
fig.show()
save_fig('flight')
#%%

dat = flight_data.Passengers
dat.index = flight_data.Month

def acorr(x, ax=None):
    x = x.values.squeeze()
    if ax is None:
        ax = plt.gca()

    x = x - x.mean()

    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[x.size:]
    autocorr /= autocorr.max()

    return ax.stem(autocorr)


fig, ax = plt.subplots()
acorr(dat[:])
ax.set_xlabel('lag(month)')
ax.set_ylabel('autocorrelation')
fig.show()


#save_fig('ac_flight')

#%%

# Correlogramm
np.random.seed(seed=0)
dat = np.random.normal(0,2,1000)
fig, axes = plt.subplots()
fig = plot_acf(dat, zero = True, lags = 40, title = 'Correlogram', ax = axes, markerfacecolor='black', color = 'black')
axes.set_xlabel("Lag")
axes.set_ylabel("ACF")
fig.show()

save_fig('correlogram', path_img2)
#%%
sm.graphics.tsa.plot_acf(dat.values.squeeze(), lags= 70)
fig.show()

save_fig('ac_flight')
#%%

fig, ax = plt.subplots() 
# fig.autofmt_xdate()
ax.plot(year_2015.Date.dt.strftime('%m-%d'),year_2015.Present_Tmax, label = 'South Korean temperature in 2015')

ax.legend(frameon=False)
ax.set_xticklabels(labels = year_2015.Date.dt.strftime('%m-%d').unique()[::3] , rotation = 45, fontsize=8)
ax.set_xticks(ax.get_xticks()[::3])
ax.set_xlabel('date')
ax.set_ylabel('temperature in Celsius')
fig.show()

#%%
# Moving average 

rolling_mean = flight_data.iloc[:,1].rolling(window = 10).mean()

fig, ax = plt.subplots() 
# fig.autofmt_xdate()
ax.plot(flight_data.Month,flight_data.Passengers, label = 'passengers')
ax.plot(flight_data.Month,rolling_mean, label = 'ma_passengers')

ax.legend(frameon=False)
ax.set_xlabel('year')
ax.set_ylabel('flight passenger volume')
fig.show()

save_fig('ma_flight_10', path_img2)
#%%

# MA correlogram passenger data
shift = 4

rolling_mean = flight_data.iloc[:,1].rolling(window = shift).mean()

error_rate = flight_data.Passengers - rolling_mean


fig, axes = plt.subplots()
fig = plot_acf(error_rate[shift:], zero = True, lags = 50, title = 'Correlogram Passenger data MA(' + str(shift-1) + ')' , ax = axes, markerfacecolor='black', color = 'black')
axes.set_xlabel("Lag")
axes.set_ylabel("ACF")
fig.show()


#save_fig('ma_flight_acf', path_img2)
#%%
# MA correlogram simulated data
shift = 4
theta = pd.DataFrame([0.8,0.6,0.4])
np.random.seed(seed=0)
wn = pd.DataFrame(np.random.normal(0,4,1000))
error_rate = pd.DataFrame(np.random.normal(10,10,1000))
for t in range(shift,1000):
    for nr, thet in enumerate(theta):
        error_rate[0][t] = thet * wn[0][t-nr]
    
fig, axes = plt.subplots()
fig = plot_acf(error_rate, zero = True, lags = 50, title = 'Correlogram simulated data MA(' + str(shift-1) + ')' , ax = axes, markerfacecolor='black', color = 'black')
axes.set_xlabel("Lag")
axes.set_ylabel("ACF")
fig.show()
save_fig('ma_simulate_acf', path_img2)
#%%
# PACF correlgram on passenger Data
fig, axes = plt.subplots()
fig = plot_pacf(flight_data.Passengers, zero = True, lags = 50, method = 'OLS', title = 'Partial correlogram on Passenger data' , ax = axes, markerfacecolor='black', color = 'black')
axes.set_xlabel("Lag")
axes.set_ylabel("ACF")
fig.show()
#%%
# PACF correlgram on simulated Data
fig, axes = plt.subplots()
fig = plot_pacf(flight_data.Passengers, zero = True, lags = 50, method = 'OLS', title = 'Partial correlogram on Passenger data' , ax = axes, markerfacecolor='black', color = 'black')
axes.set_xlabel("Lag")
axes.set_ylabel("ACF")
fig.show()

#%%
# AR model 
np.random.seed(seed=0)
dat = np.random.normal(5,8,100)
wn = np.random.normal(0,4,100)
#dat[1] = 0.7 * dat[0] + wn[0]
for nr in range(1,len(dat)):
    dat[nr] = 0.7 * dat[nr-1] + wn[nr]


ax = plt.subplot(212)   
ax.text(.5,.9,'simulated timeseries data for AR(1)', horizontalalignment = 'center',transform=ax.transAxes) 
ax = plt.plot(range(len(dat)),dat)

ax1 = plt.subplot(221)
plot_acf(dat, ax = ax1, markerfacecolor='black', color = 'black')
ax2 = plt.subplot(222)
plot_pacf(dat, zero = False, ax = ax2,markerfacecolor='black', color = 'black')

plt.savefig(path_img2 + '/ar_simulate.eps', format='eps', bbox_inches='tight')
