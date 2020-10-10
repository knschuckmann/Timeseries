#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:53:34 2020

@author: Konstantin Schuckmann
"""

import pandas as pd 
import numpy as np 
import os
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler, PowerTransformer 



'''
- habe die daten in csv umgewandelt und in eine Exle eingefügt. überwiegend mit exel die gesamt transformation gemacht
- weil daten in einem excel makro format vorlagen musste ich diese in eine für python bessere Form bringen
'''
ORIG_DATA_PATH = './timeseries/resource_data/original/Transformed Daten Beuth 2019.03.15.xlsx'
MONTH_DATA_PATH = './timeseries/resource_data/original/Monatsdaten ab 2012_fuer FC-V_in Stueck-1_CLEANED.xlsx'


def load_data(path_to_file):
    
    # create excel file document
    file = pd.ExcelFile(path_to_file)
    
    data = pd.read_excel(path_to_file, sheet_name = file.sheet_names)
    
    result_frame = []
    for key in list(data):
        result_frame.append(data[key].round(2))

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
   
    result.reset_index(level=0, inplace=True)
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

def plot_corr(data_frame, string_name, path):
    cormat = data_frame.corr()
    
    fig, ax = plt.subplots()
    im, cbar = heatmap(cormat, data_frame.columns, data_frame.columns, ax=ax,
                   cmap="RdBu", cbarlabel="correlation")
    ax.set_title('Correlation Matrix of ' + string_name[:-1])
    fig.tight_layout()
    
    save_fig(plt, string_name + 'corr', path)

def plot_hist(data_farem, string_column, string_name, path):    
    plt.hist(data_farem[string_column],density=False)
    plt.xlabel('Value Range')
    plt.ylabel('Amount')
    plt.title('Histogram of ' + string_column)
    save_fig(plt, string_name + string_column,path)
    
def create_description(data_frame_dict):
    result = {}
    for data_key ,data_frame in data_frame_dict.items():
        temp_df_main = data_frame.describe().iloc[[0,1,2,3,7],]
        temp_df_date = data_frame['Verkaufsdatum'].describe().iloc[[0,4,5],]
        temp_df_date.index = ['count', 'min', 'max']
        result.update({data_key: pd.concat([temp_df_main,temp_df_date], axis = 1)})    
    return result 

def create_description_month(data_frame_dict):
    result = {}
    for data_key ,data_frame in data_frame_dict.items():
        temp_df_main = data_frame.describe().iloc[[0,1,2,3,7],]
        result.update({data_key: temp_df_main.round(2)})    
    return result     

def create_tex_tables(dictionary, save_path_name, combined = False):    
    if combined:
        temp_dict = pd.DataFrame()
        for key, value in dictionary.items():
            temp_df = value.T
            temp_df.set_index([np.repeat(key,len(temp_df.index)), temp_df.index], inplace = True) 
            temp_dict = temp_dict.append(temp_df)
        temp_dict.to_latex(save_path_name)
    else:    
        for key, value in dictionary.items():
            value.to_latex(save_path + key + '.tex')
    

def save_fig(fig, name, path_img):
    if not os.path.exists(path_img):
        os.mkdir(path_img)
    else:
        fig.savefig(path_img + name + '.eps', format='eps', bbox_inches='tight')     
        

def create_overall_monthly(df_list):
    temp_prod = df_list[0][['Produktnummer', 'Produkt-Bezeichnung', 'PGR','PGR-Bezeichnung']]
    for num in df_list:
        temp_prod = pd.concat([temp_prod[['Produktnummer', 'Produkt-Bezeichnung', 'PGR','PGR-Bezeichnung']],num[['Produktnummer', 'Produkt-Bezeichnung', 'PGR','PGR-Bezeichnung']]])    

    temp = pd.DataFrame()
    temp[['Produktnummer', 'Produkt-Bezeichnung', 'PGR','PGR-Bezeichnung']]= temp_prod.drop_duplicates(subset=['Produktnummer'])
   
    for _ , df in enumerate(df_list) :    
        for col in df.columns[4:]:            
            if col not in temp.columns:
                temp[col] = np.zeros(len(temp['Produktnummer']))    
                
            if col in temp.columns:                
              temp[col] =  pd.merge(temp[['Produktnummer',col]],df[['Produktnummer',col]], on = 'Produktnummer', how = 'outer').iloc[:,1:3].sum(axis=1)
    return temp
    
    
def plot_datapoints_day(data_frame_1, header_string_List, range_number, savefig = False, path = './timeseries/plots/datapoints/' ):
    x, y_1, y_2 = header_string_List
    
    # range_number = 30
    # data_frame_1 = einzel_aut
    # data_frame_2 = einzel_eigVkSt
    # Header_string_List =['Verkaufsdatum', 'Einzel Menge in ST', '4Fahrt Menge in ST']
    fig, ax = plt.subplots()
    
    min_max_scaler = Normalizer()
    
    df_1 = pd.DataFrame([data_frame_1[y_1][:range_number], data_frame_1[y_2][:range_number]])
    df_1 = min_max_scaler.fit_transform(df_1) 
    
    ax.plot(data_frame_1[x][:range_number],df_1[0], label = y_1.split(' ')[0] )
    ax.plot(data_frame_1[x][:range_number],df_1[1], label = y_2.split(' ')[0] )
    
    ax.legend()
    ax.set_title(y_1.split(' ')[0] + ' and ' + y_2 + ' for ' + str(range_number) + ' datapoints')
    ax.set_ylabel('normed values')
    if savefig:
        save_fig(fig, y_1.split(' ')[0] + '_' + str(range_number) +'_datapoints',path)
    
def plot_datapoints_month(data_frame_1, list_tickets, list_time, savefig = False, path = './timeseries/plots/datapoints/' ):

    fig, ax = plt.subplots(2,1)
    
    df = data_frame_1.T
    df.columns = data_frame_1.index
    
    for d in list(df):
        if df.iloc[1,d] in list_tickets:
            ax[0].plot( np.arange(0, df.shape[0] - 4 ), df.iloc[4:,d], label = df.iloc[1,d])
    ax[0].set_title('Ticktes sold in time')
    ax[0].set_xlabel('Time')
    ax[0].legend(fontsize = 'x-small')
    # df.iloc[np.arange(4,74)]
    
    # list_plot = ['Fahrausw.Kurzstr.BLN -BO', '4 Fahrten-Karte Kurzstrecke Berlin AB', '4 Fahrten-Karte Kurzstrecke Berlin AB ermäßigt']
    normalizer = Normalizer()
    df_1 = pd.DataFrame(data_frame_1[list_time].T)
    df_1 = normalizer.fit_transform(df_1) 
    df_1 = pd.DataFrame(df_1.T, columns = list_time)

    for d in list(data_frame_1):
        if d in list_time:
            ax[1].plot(data_frame_1['Produktnummer'], df_1[d], label = d)
    ax[1].set_title('Tickets sold for each month')
    ax[1].set_xlabel('Productnumber')
    ax[1].legend()
    # ax.plot(data_frame_1[x][:range_number],df_1[1], label = y_2.split(' ')[0] )
    
    fig.tight_layout(pad=3.0)
    if savefig:
        save_fig(fig,'Monthly_datapoints',path)
    
    

def main(dayly_data = True, combine_tex = True, report_create = False):
    

    if dayly_data:
        bvg_list = load_data(ORIG_DATA_PATH)
        
        einzel_aut , einzel_eigVkSt, einzel_privat, einzel_bus, einzel_app, tages_aut, tages_eigVkSt, tages_privat, tages_bus, tages_app = bvg_list    
    
        plot_datapoints_day(einzel_aut , ['Verkaufsdatum', 'Einzel Menge in ST', '4Fahrt Menge in ST'], 365, True)
        
        overall_data = create_overall_sum_df(bvg_list.copy())
        
        names = [name.strip() for name in 'einzel_aut , einzel_eigVkSt, einzel_privat, einzel_bus, einzel_app, tages_aut, tages_eigVkSt, tages_privat, tages_bus, tages_app'.split(',')]
        
        bvg_dict = {names[i]: bvg_list[i] for i in range(len(names))} 
        bvg_dict.update({'overall_data': overall_data})

       
        descript_dict = create_description(bvg_dict)
    
        create_tex_tables(descript_dict, './timeseries/plots/latex_output/combined_dayly.tex', combined = combine_tex)
    else:        
        bvg_list = load_data(MONTH_DATA_PATH)
    
        vending_mashines, own_retailers, private_agencies, app = bvg_list 
        
        plot_datapoints_month(vending_mashines,['Fahrausw.Kurzstr.BLN -BO', '4-Fahrten-Karte', 'Tageskte.BLN AB -B1T'], [ '201301', '201710'], True)
        
        overall_data = create_overall_monthly(bvg_list)
        
        names_monthly = [name.strip() for name in 'vending_mashines, own_retailers, private_agencies, app'.split(',')]
        bvg_dict = {names_monthly[i]: bvg_list[i] for i in range(len(names_monthly))} 
        bvg_dict.update({'overall_monthly_data': overall_data})
        
        descript_month_dict = create_description_month(bvg_dict)
        
        create_tex_tables(descript_month_dict,'./timeseries/plots/latex_output/combined_monthly.tex', combined = combine_tex)
   
    if report_create:
        for dic_key, dic_entry in bvg_dict.items():
            profile = ProfileReport(dic_entry, title="Pandas Profiling Report")
            profile.to_file('./timeseries/plots/pandas_profiler/' + dic_key + '.html')
        
      
    # plot_corr(vending_mashines[col].T[1:39],'vending_mashines_', './timeseries/plots/correlation/' )
    
    #vending_mashines[vending_mashines.columns != ['Produkt-Bezeichnung', 'PGR', 'PGR-Bezeichnung'] ]
    #plot_hist(vending_mashines.iloc[:,[4:]].T, 'Einzelfahrausweise außerhalb Berlin' ,'vending_mashines_', './timeseries/plots/histograms/' )
    
#%%  
if __name__  == '__main__' : 
    main(dayly_data = False)
    