#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday September 12 2024

Set of stastical and analyitical tools

@author: william.mcm.p
"""

import pandas as pd
import os
import numpy as np
import json

# ----------------
# Stastical Tools
# ----------------

def cal_r_square(y_obs, y_fit):
    """
    Calculate the coefficient of determination (R²) for a linear regression model.

    R² represents the proportion of the variance in the observed data that is predictable from the 
    independent variables. An R² value of 1 indicates a perfect fit, while a value of 0 indicates 
    that the model explains none of the variability in the data.

    Args:
        y_obs (array-like): The observed data points (actual values).
        y_fit (array-like): The predicted data points (fitted values from the regression model).

    Returns:
        float: The R² value, indicating the goodness of fit of the model.

    Calculation:
        R² is calculated as:
        
        R² = 1 - (SS_res / SS_tot)
        
        Where:
        - SS_res (Sum of Squared Residuals) = Σ(y_obs - y_fit)²
        - SS_tot (Total Sum of Squares) = Σ(y_obs - mean(y_obs))²

    Example:
        >>> y_obs = [2, 4, 5, 8]
        >>> y_fit = [2.1, 3.9, 4.8, 7.9]
        >>> r_squared = cal_r_square(y_obs, y_fit)
        >>> print(r_squared)
        0.998
    """
    y_mean = np.mean(y_obs)                    # Mean of observed data points
    ss_res = np.sum((y_obs - y_fit) ** 2)      # Sum of squared residuals
    ss_tot = np.sum((y_obs - y_mean) ** 2)     # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)          # R² formula
    return r_squared

def normalize_data(data : pd.DataFrame, exclude : list = ['Radii(nm)'], mode = 'min-max', just_norm = False) -> pd.DataFrame:
    """
    Normalize the data in all columns except those specified in the exclude list and add new columns with normalized data.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to normalize.
        exclude (list): List of column names to exclude from normalization. Default is ['Radii(nm)'].
        mode (str): The normalization mode to apply. Options are:
                    - 'L1': Normalize data so that the sum of absolute values in each column is equal to 1.
                    - 'max': Normalize data by dividing each value by the maximum value in the column.
                    - 'min-max' (default): Normalize data to a range [0, 1] based on the minimum and maximum values in the column.
        just_norm (bool): Allows for just the noramlised dataFrame to be retuned.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for normalized data.

    Normalization Modes:
        - 'L1' Normalization: Useful when the relative proportions between data points are more important than their absolute magnitudes. 
          It scales the data such that the sum of the absolute values of each column equals 1.
        - 'max' Normalization: Scales the data relative to its maximum value, setting the maximum to 1 and scaling all other values proportionally.
          This method preserves the relative distances and distribution of the data points.
        - 'min-max' Normalization: Scales data to a fixed range [0, 1], making the minimum value 0 and the maximum value 1. It is sensitive to outliers
          because it uses the extreme values for scaling.

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]})
        >>> normalized_df = normalize_data(df, exclude=['B'], mode='max')
        Normalised data using max mode.
        >>> print(normalized_df)
           A   B  A_norm
        0  1  10   0.25
        1  2  20   0.50
        2  3  30   0.75
        3  4  40   1.00
    """
    # Stops repeated column normalistions
    columns = data.columns

    # Check if all elements in the exclude list are in columns
    for col in exclude:
        if col not in columns:
            print(f"Warning: {col} field cannot be found in provided data. Please select from {list(data.columns)}")
            return None
        
    # Clears the dataFrame to be completly empty so only the normed dataFrames a placed inside
    if just_norm:
        data = pd.DataFrame()

    for column in columns:
        # Checks if the column is not in the exclude list
        # Skips excluded columns
        if column not in exclude:

            if mode == 'L1':
                data[column + '_norm_' + mode] = data[column] / data[column].sum()

            elif mode == 'max':
                data[column + '_norm_' + mode] = data[column] / data[column].max()

            elif mode == 'all':
                max_value = data[column].max()
                min_value = data[column].min()
                data[column + '_norm_min-max'] = (data[column] - min_value) / (max_value - min_value)                    
                data[column + '_norm_max'] = data[column] / data[column].max()
                data[column + '_norm_L1'] = data[column] / data[column].sum()
                
            # The default case
            else:
                max_value = data[column].max()
                min_value = data[column].min()
                data[column + '_norm'] = (data[column] - min_value) / (max_value - min_value)                    

    print(f"Normalised data using {mode} mode.")
    return data  

def rolling_quantiles(data, window, quantiles) -> tuple[pd.Series, pd.Series]:
    """
    Calculate rolling quantiles for a given data series.

    Args:
        data (pd.Series): The input data series.
        window (int): The size of the moving window.
        quantiles (list of float): A list containing two quantiles to calculate (e.g., [0.025, 0.975]).

    Returns:
        tuple of pd.Series: The lower and upper quantiles for the rolling window.

    Example use:
        lower_quantile, upper_quantile = rolling_quantiles(data, window=100, quantiles=[0.025, 0.975])
    """
    data = pd.Series(data)
    rolling = data.rolling(window)
    return rolling.quantile(quantiles[0]), rolling.quantile(quantiles[1]).to_numpy()

def cal_average_size(data: pd.DataFrame, exclude: list = ['Radii(nm)'], target_field : str = 'Raddii(nm)') -> pd.DataFrame:
    """
    Calculate the weighted average and standard deviation for each sample in the DataFrame, excluding specified columns.

    Args:
        data (pd.DataFrame): The DataFrame containing target values and weights for each sample.
        exclude (list): List of column names to exclude from the calculation.
        target_field (str): The name of the column containing the target values.

    Returns:
        pd.DataFrame: A DataFrame with 'Sample', 'Weighted Average', and 'Weighted Std Dev' columns.
    """
    # Initialize an empty dictionary to store the average sizes
    avgs = {}
    stds = {}

    # Loop through each column in the DataFrame
    for column in data.columns:
        # Check if the column should be excluded
        if column not in exclude:
            # Calculate the weighted average size for the current column
            weights = data[column]
            weighted_avg = np.average(data[target_field], weights=weights)

            # Calculate the weighted variance
            weighted_variance = np.average((data[target_field] - weighted_avg) ** 2, weights=weights)

            # Calculate the weighted standard deviation
            weighted_std_dev = np.sqrt(weighted_variance)

            # Store the weighted average and standard deviation
            avgs[column] = weighted_avg
            stds[column] = weighted_std_dev

    # Convert the dictionary of averages to a DataFrame
    result_df = pd.DataFrame({
        'Sample': avgs.keys(),
        'Average Size (nm)': avgs.values(),
        'Standard Deviation (nm)': stds.values()
    })

    return result_df

# ----------------
# Processing Tools
# ----------------

def get_df_bins(data: pd.DataFrame, interval: int = 20) -> pd.DataFrame:
    """
    Split the DataFrame into bins based on the 'Radii (nm)' column with the given interval, 
    and calculate the mean for each bin.

    Args:
        data (pd.DataFrame): The input DataFrame.
        interval (int): The interval size for binning.

    Returns:
        pd.DataFrame: A DataFrame containing the mean values for each bin.
    """
    
    # Define the bin edges (ranges from the min to max of 'Radii (nm)' with the given interval)
    bins = np.arange(data['Radii (nm)'].min(), data['Radii (nm)'].max() + interval, interval)
    
    # Use pd.cut() to bin the 'Radii (nm)' column into intervals
    data['Bins'] = pd.cut(data['Radii (nm)'], bins=bins, right=False)
    
    # This will store the list of means for each bin
    mean_list = []

    # Finding the mean for each bin
    for bin_interval in data['Bins'].unique():
        # Get data for the current bin
        data_section = data[data['Bins'] == bin_interval]
        
        # If there's data in this bin, calculate the mean (excluding non-numeric columns like 'Bins')
        if not data_section.empty:
            bin_mean = data_section.drop(columns=['Bins']).mean()  # Exclude non-numeric columns
            # Store the bin mean
            mean_list.append(bin_mean)

    # Merging all bin means into a single DataFrame
    mean_df = pd.concat(mean_list, axis=1).T.reset_index(drop=True)
    
    # Force the first column (which is likely the 'Radii (nm)' mean) to be int
    mean_df.iloc[:, 0] = mean_df.iloc[:, 0].astype(int)

    # this will replace the radii names with the meddian values of each bin
    mean_df['Radii (nm)'] = bins[:-1] + int(interval / 2)

    return mean_df


def reorder_list(data_list):
    """
    Reorder a list so that elements ending with 'Raw' come first, 
    followed by those ending with '10Ks', '14Ks', '14Kp', and then any additional extras.

    Args:
        data_list (list of str): The list of strings to reorder.

    Returns:
        list of str: The reordered list.
    """
    # Define the order of the patterns
    order_patterns = ['Raw', '10Ks', '14Ks', '14Kp']

    # Create a list for each pattern and one for extras
    categorized = {pattern: [] for pattern in order_patterns}
    extras = []

    # Categorize each item in the data_list
    for item in data_list:
        matched = False
        for pattern in order_patterns:
            if item.endswith(pattern):
                categorized[pattern].append(item)
                matched = True
                break
        if not matched:
            extras.append(item)

    # Combine the categorized lists and extras in the defined order
    reordered_list = []
    for pattern in order_patterns:
        reordered_list.extend(categorized[pattern])
    reordered_list.extend(extras)

    return reordered_list

def group_columns(columns : list, index_field : str = 'Wavelength (nm)') -> list[list]:
    """
    Groups columns by their prefix (power and date) while including 'Wavelength (nm)' in each group.

    Args:
    - columns (list of str): List of column names.

    Returns:
    - list of list of str: A list where each element is a group of column names.
    """
    from collections import defaultdict

    # Initialize a dictionary to store grouped columns
    grouped_columns = defaultdict(list)

    if index_field not in columns:
        print(f"column {index_field} is not found in provided list {columns}")
        return None

    # Iterate through each column name
    for col in columns:
        if col == index_field:
            continue  # Skip 'Wavelength (nm)' for now
        # Extract the prefix (everything before the last hyphen-separated part)
        prefix = ' - '.join(col.split(' - ')[:2])
        grouped_columns[prefix].append(col)

    # Convert to a list of lists and include 'Wavelength (nm)' in each group
    result = [[index_field] + grouped_columns[key] for key in grouped_columns]

    return result

# the UV-Vis data had a problem where there is a noticable offset at a specific wavelength. This is due to poor calabration of the machine. 
# To resolve the problem, an offset can be use to help make the data a continous curve.
def remove_offset(data : pd.DataFrame, center_wavelength : int = 300, field : str = 'Wavelength (nm)'):
    """
    Removes an offset in UV-Vis data to correct for a noticeable correction at a specific wavelength 
    due to poor calibration of the machine, making the data a continuous curve.

    Args:
        data (pd.DataFrame): The DataFrame containing the UV-Vis data.
        center_wavelength (int): The wavelength at which the data is offset and needs correction. 
                                 Default is 300 nm.
        field (str): The column name representing the wavelength data in the DataFrame. 
                     Default is 'Wavelength (nm)'.

    Returns:
        pd.DataFrame: The corrected DataFrame with the offset applied to make the data continuous.
                      Returns None if the specified field or center wavelength is not found in the data.
    """
    # Checking if the field is present in the data provided
    if field not in data.columns:
        print(f"Warning: {field} field can not be found in prodived data. Please select from {data.columns}")
        return None

    # check that the center wavelength is present in the data
    if center_wavelength not in data[field].values:
        print(f"Warning: {center_wavelength}nm is not a wavelength present in your data")
        return None
    
    # The center point at which the data is displaced at
    center_mask =  (data[field] > center_wavelength - 3) & (data[field] < center_wavelength + 1)

    # Selecting all rows with a lower wavelength than the center wavelength
    lower_mask = data[field] < center_wavelength

    # Finds the offset difference per sample recording. 
    # The offset would naturally be negative --> may need to add logic to allow for reverse opperation if offset gets applied in wrong direction for different data points.
    offset = np.diff(np.diff(data[center_mask], axis=0), axis=0)

    # Apply the offset to all required rows
    data[lower_mask] += offset

    return data