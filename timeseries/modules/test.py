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