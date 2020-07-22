#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:53:34 2020

@author: Konstantin Schuckmann
"""

import pandas as pd 
import numpy as np 
from pandas_profiling import ProfileReport


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



def create_overall_sum_df(data):
    
    for nr, dat in enumerate(data):
        data[nr] = dat.set_index('Verkaufsdatum')
        
    result = data[0]
    result['Tages Menge in ST'] = np.zeros(len(result))
    result['Tages Wert in EUR'] = np.zeros(len(result))

    for _, dat in enumerate(data[1:]):
        if dat.columns.str.match('Einzel').any() == True:
            result['Einzel Menge in ST'] = result['Einzel Menge in ST'].add(dat['Einzel Menge in ST'], fill_value = 0)  
            result['Einzel Wert in EUR'] = result['Einzel Wert in EUR'].add(dat['Einzel Wert in EUR'], fill_value = 0)
            if dat.columns.str.match('4Fahrt').any() == True:
                result['4Fahrt Menge in ST'] = result['4Fahrt Menge in ST'].add(dat['4Fahrt Menge in ST'], fill_value = 0)  
                result['4Fahrt Wert in EUR'] = result['4Fahrt Wert in EUR'].add(dat['4Fahrt Wert in EUR'], fill_value = 0)  
        else:
            result['Tages Menge in ST'] = result['Tages Menge in ST'].add(dat['Tages Menge in ST'], fill_value = 0)  
            result['Tages Wert in EUR'] = result['Tages Wert in EUR'].add(dat['Tages Wert in EUR'], fill_value = 0)  
            
        result['Gesamt Menge in ST'] = result['Gesamt Menge in ST'].add(dat['Gesamt Menge in ST'], fill_value = 0)  
        result['Gesamt Wert in EUR'] = result['Gesamt Wert in EUR'].add(dat['Gesamt Wert in EUR'], fill_value = 0)  
    
    return result

def main():
    
    data = load_data(ORIG_DATA_PATH)
    
    overall_data = create_overall_sum_df(data)
    
    einzel_aut , einzel_eigVkSt, einzel_privat, einzel_bus, einzel_app, tages_aut, tages_eigVkSt, tages_privat, tages_bus, tages_app = data    
   
    
    
    profile = ProfileReport(einzel_aut, title="Pandas Profiling Report")
    
    profile.to_file("your_report.html")
    
if __name__  == '__main__' : 
    main()
    