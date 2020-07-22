#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:53:34 2020

@author: Konstantin Schuckmann
"""

import pandas as pd 
import numpy as np 

import plotly
import plotly.graph_objs as go

'''
- habe die daten in csv umgewandelt und in eine Exle eingefügt. überwiegend mit exel die gesamt transformation gemacht
- weil daten in einem excel makro format vorlagen musste ich diese in eine für python bessere Form bringen
'''
ORIG_DATA_PATH = './timeseries/resource_data/original/Transformed Daten Beuth 2019.03.15.xlsx'


def load_data(ORIG_DATA_PATH):
    
    # create excel file document
    file = pd.ExcelFile(ORIG_DATA_PATH)
    
    data = pd.read_excel(ORIG_DATA_PATH, sheet_name = file.sheet_names)
    
    result_frame = []
    for key in list(data):
        result_frame.append(data[key])
    
    return result_frame


def check_for_nan(data):
    
    result={}
    for nr, df in enumerate(data):
        if df.isnull().values.any():
            result.update({nr:True})
    
    if not result:
        result = False
    
    return result


import plotly.express as px
df = einzel_aut
#fig = px.scatter_3d(df, x='Verkaufsdatum', y='Einzel Wert in EUR', z='Einzel Menge in ST')
fig = px.histogram(df, x="Einzel Wert in EUR") # histogram
fig = px.histogram(df, x="Einzel Menge in ST")
plotly.offline.plot(fig, filename = 'test')

    
def main():
    
    data = load_data(ORIG_DATA_PATH)
    
    
    einzel_aut = data[0]
    einzel_aut
    einzel_eigVkSt = data[1] 
    einzel_privat = data[2] 
    einzel_bus = data[3] 
    einzel_app = data[4] 
    tages_aut = data[5] 
    tages_eigVkSt = data[6] 
    tages_privat = data[7] 
    tages_bus = data[8] 
    tages_app = data[9] 
    
    
    
    
    
    
if __name__  == '__main__' : 
    main()
    