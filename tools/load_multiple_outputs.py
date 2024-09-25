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

from src.DataLoader import load_multiple_outputs, save_dataframe, remove_offset, normalize_data, bulk_dump_columns

# Loading the multiple files and then saving to a parquet file
directory = "/Users/william/Library/CloudStorage/OneDrive-SwinburneUniversity/Classes/2024 hons/Silicon NP and Optical Forces - Project/Centrifugation Data/2024/20240924/UV-Vis Spectrums/"

merged_column_name = 'Wavelength (nm)'

loaded_data = load_multiple_outputs(directory, merged_column_name, '.csv', col_formatter='')

# Changeing the name of the column 0 -> 0.6W - Raw 
# Cause fromthe load_multiple_output() name strings processing
loaded_data = loaded_data.rename(columns={'0': '0.6W - Raw'})

order = ['Wavelength (nm)', '0.6W - Raw', '1W - Raw', '2W - Raw', '3W - Raw', '4W - Raw',
        '5W  - Raw', '6W - Raw', '10W - Raw']

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

# Normalise the data using min-max mathod
loaded_data = normalize_data(loaded_data, ['Wavelength (nm)'])

# loaded_data_filtered = loaded_data[loaded_data['Wavelength (nm)'] <= 1100]

# Save the Data frame as parquet
path = "/Users/william/Library/CloudStorage/OneDrive-SwinburneUniversity/Classes/2024 hons/Silicon NP and Optical Forces - Project/Centrifugation Data/2024/20240924/UV-Vis Spectrums.parquet"
print("Saving the UV_Vis_spectrum df")
# save_dataframe(loaded_data, path)


# Bulk dump each column into a txt file for the optmisation technique
# print("Column dummping UV_Vis_spectrum df - filtered (Wavelength (nm)' <= 900) -> for best optmisation results")
# bulk_dump_columns(loaded_data_filtered, directory)
