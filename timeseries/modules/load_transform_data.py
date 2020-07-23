#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:53:34 2020

@author: Konstantin Schuckmann
"""

import pandas as pd 
import numpy as np 
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt


'''
- habe die daten in csv umgewandelt und in eine Exle eingefügt. überwiegend mit exel die gesamt transformation gemacht
- weil daten in einem excel makro format vorlagen musste ich diese in eine für python bessere Form bringen
'''
ORIG_DATA_PATH = './timeseries/resource_data/original/Transformed Daten Beuth 2019.03.15.xlsx'


def load_data(path_to_file):
    
    # create excel file document
    file = pd.ExcelFile(path_to_file)
    
    data = pd.read_excel(path_to_file, sheet_name = file.sheet_names)
    
    result_frame = []
    for key in list(data):
        result_frame.append(data[key])

    return result_frame



def create_overall_sum_df(given_data):
    
    for nr, dat in enumerate(given_data):
        given_data[nr] = dat.set_index('Verkaufsdatum')
        
    result = given_data[0]
    result['Tages Menge in ST'] = np.zeros(len(result))
    result['Tages Wert in EUR'] = np.zeros(len(result))

    for _, dat in enumerate(given_data[1:]):
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

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, vmin= -1, vmax = 1, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # # Turn spines off and create white grid.
    # for edge, spine in ax.spines.items():
    #     spine.set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def plot_corr(data_frame):
    cormat = data_frame.corr()
    
    fig, ax = plt.subplots()
    im, cbar = heatmap(cormat, data_frame.columns, data_frame.columns, ax=ax,
                   cmap="RdBu", cbarlabel="test")
    fig.tight_layout()
    plt.show()

def plot_hist(data_farem, string_name):    
    plt.hist(data_farem[string_name],density=False)
    plt.xlabel('Value Range')
    plt.ylabel('Amount')
    plt.title('Histogram of ' + string_name)
    plt.show()
    

def main():
    
    bvg_orig_list = load_data(ORIG_DATA_PATH)
    
    einzel_aut , einzel_eigVkSt, einzel_privat, einzel_bus, einzel_app, tages_aut, tages_eigVkSt, tages_privat, tages_bus, tages_app = bvg_orig_list    

    overall_data = create_overall_sum_df(bvg_orig_list.copy())
    
    names = [name.strip() for name in 'einzel_aut , einzel_eigVkSt, einzel_privat, einzel_bus, einzel_app, tages_aut, tages_eigVkSt, tages_privat, tages_bus, tages_app'.split(',')]
    
    bvg_orig_dict = {names[i]: bvg_orig_list[i] for i in range(len(names))} 
    bvg_orig_dict.update({'overall_data': overall_data})

    for dic_key, dic_entry in bvg_orig_dict.items():
        profile = ProfileReport(dic_entry, title="Pandas Profiling Report")
        profile.to_file('./timeseries/plots/pandas_profiler' + dic_key + '.html')
    
    plot_corr(overall_data)
    
    plot_hist(overall_data, 'Gesamt Menge in ST')
    
    
    
    
if __name__  == '__main__' : 
    main()
    