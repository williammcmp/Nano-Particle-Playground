#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday September 28 2024

Takes all files found in a given folder that matches a specific pattern, 
load the data into a combined DataFrame and then saves the data into a parquet file type for accessing latter

@author: william.mcm.p
"""
# Loading in the Libraries
import numpy as np
import pandas as pd

from DataLoader import load_multiple_outputs, save_dataframe, remove_offset, normalize_data, bulk_dump_columns

con_path = "E:/Users/William/Uni/Swinburne OneDrive/OneDrive - Swinburne University/Classes/2024 hons/Silicon NP and Optical Forces - Project/Centrifugation Data/2024/20240924/Optmisation outputs/concentrations/"
spec_path = "E:/Users/William/Uni/Swinburne OneDrive/OneDrive - Swinburne University/Classes/2024 hons/Silicon NP and Optical Forces - Project/Centrifugation Data/2024/20240924/Optmisation outputs/fitted_spectrums/"


con_data = load_multiple_outputs(con_path, 'Radii (nm)', '.txt', col_formatter = ".")
spec_data = load_multiple_outputs(spec_path, 'Wavelength (nm)', '.txt', col_formatter='.')

# Changeing the name of the column 0 -> 0.6W - Raw 
# Cause fromthe load_multiple_output() name strings processing
con_data = con_data.rename(columns={'0-6W - Raw': '0.6W - Raw',
                                          '0-6W - 5kp': '0.6W - 5kp', 
                                          '0-6W - 5ks': '0.6W - 5ks',
                                          '0-6W - 5kp5ks': '0.6W - 5kp5ks',
                                          '0-6W - 5kp5kp5ks': '0.6W - 5kp5kp5ks',
                                          '0-6W - 5kp5kp5kp': '0.6W - 5kp5kp5kp'})

spec_data = spec_data.rename(columns={'0-6W - Raw': '0.6W - Raw',
                                          '0-6W - 5kp': '0.6W - 5kp', 
                                          '0-6W - 5ks': '0.6W - 5ks',
                                          '0-6W - 5kp5ks': '0.6W - 5kp5ks',
                                          '0-6W - 5kp5kp5ks': '0.6W - 5kp5kp5ks',
                                          '0-6W - 5kp5kp5kp': '0.6W - 5kp5kp5kp'})

# re_ordering the columns
con_data = con_data[['Radii (nm)', 
        '0.6W - Raw', '0.6W - 5ks', '0.6W - 5kp', '0.6W - 5kp5ks', '0.6W - 5kp5kp5ks', '0.6W - 5kp5kp5kp',
        '1W - Raw', '1W - 5ks', '1W - 5kp', '1W - 5kp5ks', '1W - 5kp5kp5ks', '1W - 5kp5kp5kp',
        '2W - Raw', '2W - 5ks', '2W - 5kp', '2W - 5kp5ks', '2W - 5kp5kp5ks', '2W - 5kp5kp5kp',
        '3W - Raw', '3W - 5ks', '3W - 5kp', '3W - 5kp5ks', '3W - 5kp5kp5ks', '3W - 5kp5kp5kp',
        '4W - Raw', '4W - 5ks', '4W - 5kp', '4W - 5kp5ks', '4W - 5kp5kp5ks', '4W - 5kp5kp5kp',
        '5W - Raw', '5W - 5ks', '5W - 5kp',                '5W - 5kp5kp5ks', '5W - 5kp5kp5kp',
        '6W - Raw', '6W - 5ks', '6W - 5kp', '6W - 5kp5ks', '6W - 5kp5kp5ks', '6W - 5kp5kp5kp',
        '10W - Raw', '10W - 5ks', '10W - 5kp', '10W - 5kp5ks', '10W - 5kp5kp5ks', '10W - 5kp5kp5kp',
        ]]


spec_data = spec_data[['Wavelength (nm)', 
        '0.6W - Raw', '0.6W - 5ks', '0.6W - 5kp', '0.6W - 5kp5ks', '0.6W - 5kp5kp5ks', '0.6W - 5kp5kp5kp',
        '1W - Raw', '1W - 5ks', '1W - 5kp', '1W - 5kp5ks', '1W - 5kp5kp5ks', '1W - 5kp5kp5kp',
        '2W - Raw', '2W - 5ks', '2W - 5kp', '2W - 5kp5ks', '2W - 5kp5kp5ks', '2W - 5kp5kp5kp',
        '3W - Raw', '3W - 5ks', '3W - 5kp', '3W - 5kp5ks', '3W - 5kp5kp5ks', '3W - 5kp5kp5kp',
        '4W - Raw', '4W - 5ks', '4W - 5kp', '4W - 5kp5ks', '4W - 5kp5kp5ks', '4W - 5kp5kp5kp',
        '5W - Raw', '5W - 5ks', '5W - 5kp',                '5W - 5kp5kp5ks', '5W - 5kp5kp5kp',
        '6W - Raw', '6W - 5ks', '6W - 5kp', '6W - 5kp5ks', '6W - 5kp5kp5ks', '6W - 5kp5kp5kp',
        '10W - Raw', '10W - 5ks', '10W - 5kp', '10W - 5kp5ks', '10W - 5kp5kp5ks', '10W - 5kp5kp5kp',
        ]]

out_path = 'E:/Users/William/Uni/Swinburne OneDrive/OneDrive - Swinburne University/Classes/2024 hons/Silicon NP and Optical Forces - Project/Centrifugation Data/2024/20240924/Optmisation outputs/'

# Saving the data to parquet files
save_dataframe(con_data, out_path + "concentractions.parquet")
save_dataframe(spec_data, out_path + "fitted_spectrums.parquet")