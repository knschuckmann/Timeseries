#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:03:51 2020

@author: Konstantin Schuckmann
"""
############################################################################################# 
############################################ DUMMY ##########################################
#############################################################################################
dummy_temperature_path = '../resource_data/dummy/Bias_correction_ucl.csv' 
dummy_flight_path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
dummy_save_path_section_2 = '../../../../Latex/bhtThesis/Masters/2_fundamentals/pictures'
dummy_save_path_section_3 = '../../../../Latex/bhtThesis/Masters/3_used_Models/pictures/'


############################################################################################# 
################################# LOAD AND TRANSFORM DATA ###################################
#############################################################################################
ORIG_DATA_PATH = '../resource_data/original/Transformed Daten Beuth 2019.03.15.xlsx'
MONTH_DATA_PATH = '../resource_data/original/Monatsdaten ab 2012_fuer FC-V_in Stueck-1_CLEANED.xlsx'
DATA = 'combined_df'
SAVE_PLOTS_PATH = '../../../../Latex/bhtThesis/Masters/5_evaluation/pictures/more_steps/Gesamt/' + DATA + '/'
SAVE_MODELS_PATH = '../resource_data/models/more_steps/Gesamt/' + DATA + '/'
SAVE_RESULTS_PATH = '../resource_data/models/results/'