#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday September 25 2024

Takes all files found in a given folder that matches a specific pattern, 
load the data into a combined DataFrame and then saves the data into a parquet file type for accessing latter

@author: william.mcm.p
"""
# Loading in the Libraries
import numpy as np
import pandas as pd

from DataLoader import load_multiple_outputs, save_dataframe, remove_offset, normalize_data, bulk_dump_columns

# Loading the multiple files and then saving to a parquet file
directory = "E:/Users/William/Uni/Swinburne OneDrive/OneDrive - Swinburne University/Classes/2024 hons/Silicon NP and Optical Forces - Project/Centrifugation Data/2024/20240924/UV-Vis Spectrums/"


merged_column_name = 'Wavelength (nm)'

loaded_data = load_multiple_outputs(directory, merged_column_name, '.csv', col_formatter='.')

print(loaded_data.columns)

# Changeing the name of the column 0 -> 0.6W - Raw 
# Cause fromthe load_multiple_output() name strings processing
loaded_data = loaded_data.rename(columns={'0-6W - Raw': '0.6W - Raw',
                                          '0-6W - 5kp': '0.6W - 5kp', 
                                          '0-6W - 5ks': '0.6W - 5ks',
                                          '0-6W - 5kp5ks': '0.6W - 5kp5ks',
                                          '0-6W - 5kp5kp5ks': '0.6W - 5kp5kp5ks',
                                          '0-6W - 5kp5kp5kp': '0.6W - 5kp5kp5kp'})
# Changing the order of the data files
order = ['Wavelength (nm)', 
        '0.6W - Raw', '0.6W - 5ks', '0.6W - 5kp', '0.6W - 5kp5ks', '0.6W - 5kp5kp5ks', '0.6W - 5kp5kp5kp',
        '1W - Raw', '1W - 5ks', '1W - 5kp', '1W - 5kp5ks', '1W - 5kp5kp5ks', '1W - 5kp5kp5kp',
        '2W - Raw', '2W - 5ks', '2W - 5kp', '2W - 5kp5ks', '2W - 5kp5kp5ks', '2W - 5kp5kp5kp',
        '3W - Raw', '3W - 5ks', '3W - 5kp', '3W - 5kp5ks', '3W - 5kp5kp5ks', '3W - 5kp5kp5kp',
        '4W - Raw', '4W - 5ks', '4W - 5kp', '4W - 5kp5ks', '4W - 5kp5kp5ks', '4W - 5kp5kp5kp',
        '5W - Raw', '5W - 5ks', '5W - 5kp',                '5W - 5kp5kp5ks', '5W - 5kp5kp5kp',
        '6W - Raw', '6W - 5ks', '6W - 5kp', '6W - 5kp5ks', '6W - 5kp5kp5ks', '6W - 5kp5kp5kp',
        '10W - Raw', '10W - 5ks', '10W - 5kp', '10W - 5kp5ks', '10W - 5kp5kp5ks', '10W - 5kp5kp5kp',
        ]

loaded_data = loaded_data[order]


# # Interploating the data to be at 1nm steps
wavelengths = np.arange(loaded_data['Wavelength (nm)'].min(), loaded_data['Wavelength (nm)'].max() + 1)
loaded_data.set_index('Wavelength (nm)', inplace=True)

# Reindex with new wavelength steps and interpolate missing values
loaded_data = loaded_data.reindex(wavelengths).interpolate()

# Reset index to make Wavelength a column again
loaded_data.reset_index(inplace=True)
loaded_data.rename(columns={'index': 'Wavelength (nm)'}, inplace=True)
# # Interploation Completed

loaded_data = remove_offset(loaded_data, 311, scale=1)

# # Normalise the data using min-max mathod
loaded_data = normalize_data(loaded_data, ['Wavelength (nm)'])

loaded_data_filtered = loaded_data[loaded_data['Wavelength (nm)'] <= 1100]

# Save the Data frame as parquet
path = "E:/Users/William/Uni/Swinburne OneDrive/OneDrive - Swinburne University/Classes/2024 hons/Silicon NP and Optical Forces - Project/Centrifugation Data/2024/20240924/UV-Vis Spectrums_complete.parquet"
print("Saving the UV_Vis_spectrum df")
# save_dataframe(loaded_data, path)


# Bulk dump each column into a txt file for the optmisation technique
print("Column dummping UV_Vis_spectrum df - filtered (Wavelength (nm)' <= 900) -> for best optmisation results")
# bulk_dump_columns(loaded_data_filtered, directory + "cleaned_spectrums/")