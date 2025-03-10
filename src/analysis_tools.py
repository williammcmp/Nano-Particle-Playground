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

def get_trend_fit(x_data : pd.DataFrame, y_data : pd.DataFrame, degree : int = 1):
    """
    Calculate the best-fit trend line or curve for the given x and y data using polynomial regression.

    This function fits a polynomial of the specified degree to the input data and returns the fitted 
    values (y-values) based on the best-fit line or curve.

    Args:
        x_data (pd.DataFrame): The input data for the x-axis (independent variable).
        y_data (pd.DataFrame): The input data for the y-axis (dependent variable).
        degree (int, optional): The degree of the polynomial to fit. Default is 1 (linear regression).

    Returns:
        pd.Series: The y-values of the best-fit trend line or curve for the input x_data.

    Notes:
        - For a linear regression (degree = 1), the function returns the y-values of the best-fit line.
        - For higher-degree polynomials, the function returns the y-values of the best-fit polynomial curve.
        - The input data is converted to numeric form if necessary, to ensure proper calculations.

    Example:
        >>> x_data = pd.Series([1, 2, 3, 4, 5])
        >>> y_data = pd.Series([2, 4, 5, 7, 10])
        >>> y_fit = get_trend_fit(x_data, y_data, degree=1)
        >>> print(y_fit)

    """
    # Converting the data into numerical incase it is not already
    x_data = pd.to_numeric(x_data.copy())
    y_data = pd.to_numeric(y_data.copy())

    slope, intercept = np.polyfit(x_data, y_data, degree)

    y_fit = slope * x_data + intercept

    return y_fit

def normalize_data(data_in : pd.DataFrame, exclude : list = ['Radii(nm)'], mode = 'min-max') -> pd.DataFrame:
    """
    Normalize the data in all columns except those specified in the exclude list and add new columns with normalized data.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to normalize.
        exclude (list): List of column names to exclude from normalization. Default is ['Radii(nm)'].
        mode (str): The normalization mode to apply. Options are:
                    - 'L1': Normalize data so that the sum of absolute values in each column is equal to 1.
                    - 'max': Normalize data by dividing each value by the maximum value in the column.
                    - 'min-max' (default): Normalize data to a range [0, 1] based on the minimum and maximum values in the column.

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
    data = data_in.copy()

    # Stops repeated column normalistions
    columns = data.columns

    # Check if all elements in the exclude list are in columns
    for col in exclude:
        if col not in columns:
            print(f"Warning: {col} field cannot be found in provided data. Please select from {list(data.columns)}")
            return None

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


def get_mass_precent(data_in : pd.DataFrame, norm_mode = 'L1') -> pd.DataFrame:
    """
    Calculate the mass percentage for each particle size in the given DataFrame, and optionally normalize the data.

    This function computes the mass of silicon particles based on their radii (given in nanometers) and the known 
    density of silicon (2330 kg/m^3). The mass is calculated for each size, and the input data is scaled accordingly.
    Optionally, the data can be normalized using the specified normalization mode.

    Args:
        data_in (pd.DataFrame): The input DataFrame containing the particle size data (column 'Radii (nm)') and 
                                corresponding raw measurements for each sample.
        norm_mode (str, optional): The normalization mode to use. If provided, the data will be normalized. 
                                   The default is 'L1'. If set to `None`, no normalization is applied.

    Returns:
        pd.DataFrame: A DataFrame containing the mass percentage of the samples. If normalization is applied, 
                      the '_norm' suffix is removed from the column names. The 'Radii (nm)' column is retained 
                      in the output.

    Steps:
        1. Compute the volume of each particle assuming spherical particles.
        2. Compute the mass of each particle using the volume and the density of silicon (2330 kg/m^3).
        3. Scale the sample data in the DataFrame by the calculated mass for each particle size.
        4. Optionally normalize the data using the specified `norm_mode`.
        5. Return the processed DataFrame with the mass percentage for each sample and the particle size ('Radii (nm)').

    Example:
        >>> data_in = pd.DataFrame({
        ...     'Radii (nm)': [10, 20, 30],
        ...     'Sample1': [0.1, 0.2, 0.3],
        ...     'Sample2': [0.4, 0.5, 0.6]
        ... })
        >>> result = get_mass_precent(data_in)
        >>> print(result)

    Notes:
        - The function assumes spherical particles for volume calculation.
        - The default normalization mode is 'L1'. Other modes can be specified, or set to `None` to skip normalization.
        - The output DataFrame retains the 'Radii (nm)' column and removes any temporary columns (e.g., 'Bins') if present.

    """

    data = data_in.copy()

    volume = (4/3) * np.pi * np.pow((data_in['Radii (nm)'] * 1e-9), 3) # volume in SI --> m^3

    silicon_density = 2330 #kg/m^3

    mass = volume * silicon_density

    # Drops the Radius column -> not needed for the analysis
    data = data.drop('Radii (nm)', axis = 1)

    # Removes the Bins field if present
    if 'Bins' in data.columns:
        data = data.drop('Bins', axis = 1)
    

    for y_col in data.columns:
        data[y_col] = data[y_col] * mass


    data['Radii (nm)'] = data_in['Radii (nm)']

    # Will normalise the data if a mode is proided
    if norm_mode is not None:
        data = normalize_data(data, ['Radii (nm)'], mode = norm_mode)

        # Only return the norm values
        data = data.filter(like='_norm')

    # Remove the _norm* from each of the column names
    data.columns = data.columns.str.replace(r'_.*', '', regex=True)


    # Add the radii column back to the df
    data['Radii (nm)'] = data_in['Radii (nm)']

    return data

# ----------------
# Processing Tools
# ----------------

def get_df_bins(data: pd.DataFrame, interval: int = 20, bin_edges = None) -> pd.DataFrame:
    """
    Split the DataFrame into bins based on the 'Radii (nm)' column with the given interval, 
    and calculate the mean for each bin.

    Args:
        data (pd.DataFrame): The input DataFrame.
        interval (int): The interval size for binning.

    Returns:
        pd.DataFrame: A DataFrame containing the mean values for each bin.
    """
    
    if bin_edges is not None:
        bins = bin_edges
    else:
        # Define the bin edges (ranges from the min to max of 'Radii (nm)' with the given interval)
        bins = np.arange(data['Radii (nm)'].min(), data['Radii (nm)'].max() + interval, interval)
    
    print(bins)
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
            bin_mean = data_section.drop(columns=['Bins']).sum()  # Exclude non-numeric columns
            # Store the bin mean
            mean_list.append(bin_mean)

    # Merging all bin means into a single DataFrame
    mean_df = pd.concat(mean_list, axis=1).T.reset_index(drop=True)
    
    # Force the first column (which is likely the 'Radii (nm)' mean) to be int
    mean_df.iloc[:, 0] = mean_df.iloc[:, 0].astype(int)

    # this will replace the radii names with the meddian values of each bin
    mean_df['Radii (nm)'] = bins[:-1] + np.round(interval / 2, 0)

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

def group_columns(columns : list, index_field : str = 'Wavelength (nm)', index : int = 2) -> list[list]:
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
        prefix = ' - '.join(col.split(' - ')[index:])
        grouped_columns[prefix].append(col)

    # Convert to a list of lists and include 'Wavelength (nm)' in each group
    result = [[index_field] + grouped_columns[key] for key in grouped_columns]

    return result

# the UV-Vis data had a problem where there is a noticable offset at a specific wavelength. This is due to poor calabration of the machine. 
# To resolve the problem, an offset can be use to help make the data a continous curve.
def remove_offset(data : pd.DataFrame, center_wavelength : int = 300, field : str = 'Wavelength (nm)', scale=1):
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
    offset = np.diff(np.diff(data[center_mask], axis=0), axis=0) * scale

    # Apply the offset to all required rows
    data[lower_mask] += offset

    print(f'Rmoved {offset.min():.2f} offset at {center_wavelength} nm')

    return data

def remove_str_from_cols(data: pd.DataFrame, str_pattern: str = '_norm', hold_list: list = []):
    """
    Filter columns based on a string pattern, remove the matching part from the column names,
    and retain specific columns unmodified, placing them at the front of the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame to operate on.
        str_pattern (str, optional): String pattern to filter columns. Default is '_norm'.
        hold_list (list, optional): List of column names to be held and added back to the DataFrame 
                                    after modification, placed at the front of the DataFrame.
                                    Default is an empty list.

    Returns:
        pd.DataFrame: A new DataFrame with columns filtered and renamed, while holding specific columns 
                      unmodified and placing them at the front of the DataFrame.
    """
    data = data.copy()  # Defensive to avoid modifying the original object

    # Check if all columns in hold_list exist in the original DataFrame
    for col in hold_list:
        if col not in data.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")
    
    # Temporarily hold the specified columns
    hold_df = data[hold_list].copy()

    # Filter and rename the columns based on the str_pattern
    data = data.filter(like=str_pattern)

    # Remove the matching pattern and anything after it in the column names
    data.columns = data.columns.str.replace(r'_.*', '', regex=True)

    # Re-add the hold_list columns to the DataFrame
    data[hold_list] = hold_df

    # Reorder the columns to put hold_list at the front
    cols = hold_list + [col for col in data.columns if col not in hold_list]
    data = data[cols]

    return data

