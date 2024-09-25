#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday September 12 2024

Function and script use to load and save data

@author: william.mcm.p
"""


import pandas as pd
import os
import numpy as np
import json
from src.analysis_tools import *

def load_data_from_csv(file_path : str):
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing the data from the CSV file.
            Returns None if the file is not found or an error occurs during loading.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from CSV file: {str(e)}")
        return None
    
def load_data_from_txt(file_path : str, header : list[str] = None) -> pd.DataFrame:
    """
    Load data from a .txt file.

    Args:
        file_path (str): The path to the .txt file.

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing the data from the .txt file.
            Returns None if the file is not found or an error occurs during loading.
    """
    try:
        data = pd.read_csv(file_path, sep='\t', header=header)
        return data
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from .txt file: {str(e)}")
        return None
    

def load_data_from_excel(file_path, sheet_name=0, load_all_sheets=False):
    """
    Load data from an Excel file.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str or int, optional): The sheet name or index to load.
            Defaults to 0 (the first sheet).
        load_all_sheets (bool, optional): If True, loads all sheets and returns
            a dictionary of DataFrames. Defaults to False.

    Returns:
        pd.DataFrame or dict or None: A pandas DataFrame containing the data from the Excel file
            if load_all_sheets is False, or a dictionary of DataFrames if load_all_sheets is True.
            Returns None if the file is not found or an error occurs during loading.
    """
    try:
        if load_all_sheets:
            data = pd.read_excel(file_path, sheet_name=None)
        else:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
        return data
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from Excel file: {str(e)}")
        return None

def get_file_names(folder_path : str, filter_pattern : str = None):
    """
    Get a list of filenames in the specified folder path.

    Args:
        folder_path (str): The path to the folder from which to retrieve filenames.
        filter_pattern (str): A pattern to filter the list of filenames by --> recommended to use file types

    Returns:
        list of str: A list of filenames in the specified folder.
    """

    file_names = os.listdir(folder_path)

    # filter out the list of filenames based on a pattern
    if filter_pattern is not None:
        file_names = [col for col in file_names if filter_pattern in col.lower()]

    return file_names

# Adjust values depending on what side of the crator they are recorded from
def experimental_adjustment(experimental_csv : str, experimental_data :  pd.DataFrame, data_df : pd.DataFrame):
    """
    Adjust and append experimental data to a combined DataFrame based on the position relative to the ablation creator.

    Args:
        experimental_csv (str): The name of the experimental CSV file.
        experimental_data (pd.DataFrame): The experimental data loaded from the CSV file.
        data_df (pd.DataFrame): The combined data DataFrame to which the experimental data will be appended.

    Returns:
        pd.DataFrame: The updated combined data DataFrame.
    """
    
    # Checks the position relative to the ablation creator the data is from

    # Positive X-axis
    if experimental_csv[0] == "R": # Right of the creator
        
        # Create a new DataFrame for the data
        data = pd.DataFrame(data=experimental_data, columns=["X", "Y", "Width"])
        
        # Modify the 'Y' values
        data["Y"] *= 0
        
        # Increasing the distance from the creator for further images
        max_value = data_df["X"].max() # max value of the whole data set

        # Account for the missing Right1 values for BFieldIn
        if max_value == 0 and experimental_csv[5] == 2:
            max_value += 2285

        # Increasing by the pervious image's max positional value
        data['X'] = data['X'] + max_value

        # Append the data to the combined data frame
        data_df = pd.concat([data_df, data], ignore_index=True)

    # Negative x-axis 
    elif experimental_csv[0] == "L": # Left of the crator
        # Create a new DataFrame for the data
        data = pd.DataFrame(data=experimental_data, columns=["X", "Y", "Width"])
        
        # Set Values to zero. 
        data["Y"] *= 0
        
        # Increasing the distance from the creator for further images
        max_value = data["X"].max() # max value of the current image
        min_value = data_df["X"].min() # min value of the whole dataset


        # Increasing by the pervious image's max positional value
        data['X'] = min_value - (max_value - data['X'])


        # Append the data to the combined data frame
        data_df = pd.concat([data_df, data], ignore_index=True)

    # Positive Y-axis
    elif experimental_csv[0] == "T": # Above the creator
        
        # Create a new DataFrame for the data
        data = pd.DataFrame(data=experimental_data, columns=["X", "Y", "Width"])
        
        # Modify the 'Y' values
        data["X"] *= 0
        
        # Increasing the distance from the creator for further images
        max_value = data_df["Y"].max() # max value of the whole data set

        # Increasing by the pervious image's max positional value
        data['Y'] = data['Y'] + max_value

        # Append the data to the combined data frame
        data_df = pd.concat([data_df, data], ignore_index=True)

    # Negative Y-axis 
    elif experimental_csv[0] == "B": # Bellow the crator
        # Create a new DataFrame for the data
        data = pd.DataFrame(data=experimental_data, columns=["X", "Y", "Width"])
        
        # Set Values to zero. 
        data["X"] *= 0
        
        # Increasing the distance from the creator for further images
        max_value = data["Y"].max() # max value of the current image
        min_value = data_df["Y"].min() # min value of the whole dataset


        # Increasing by the pervious image's max positional value
        data['Y'] = min_value - (max_value - data['Y'])

        # Append the data to the combined data frame
        data_df = pd.concat([data_df, data], ignore_index=True)

    return data_df



def Experimental_special_adjustment(experimental_csv : str, experimental_data :  pd.DataFrame, data_df : pd.DataFrame):

    # Checks the position relative to the ablation creator the data is from

    # Positive X-axis
    if experimental_csv.find("R") != -1: # Right IRL, but move above the creator in the SEM
        
        # Create a new DataFrame for the data
        data = pd.DataFrame(data=experimental_data, columns=["X", "Y", "Width"])
        
        # Modify the 'X' values
        data["X"] *= 0
        
        # Increasing the distance from the creator for further images
        max_value = data_df["Y"].max() # max value of the whole data set


        # Increasing by the pervious image's max positional value
        data['Y'] = data['Y'] + max_value

        # Append the data to the combined data frame
        data_df = pd.concat([data_df, data], ignore_index=True)

    # Negative x-axis 
    elif experimental_csv.find("L")  != -1: # Left of the crator
        # Create a new DataFrame for the data
        data = pd.DataFrame(data=experimental_data, columns=["X", "Y", "Width"])
        
        # Set Values to zero. 
        data["X"] *= 0
        
        # Increasing the distance from the creator for further images
        max_value = data["Y"].max() # max value of the current image
        min_value = data_df["Y"].min() # min value of the whole dataset


        # Increasing by the pervious image's max positional value
        data['Y'] = min_value - (max_value - data['Y'])


        # Append the data to the combined data frame
        data_df = pd.concat([data_df, data], ignore_index=True)

    # Positive Y-axis
    elif experimental_csv.find("B")  != -1: # Above the creator
        
        # Create a new DataFrame for the data
        data = pd.DataFrame(data=experimental_data, columns=["X", "Y", "Width"])
        
        # Modify the 'Y' values
        data["Y"] *= 0
        
        # Increasing the distance from the creator for further images
        max_value = data_df["X"].max() # max value of the whole data set

        # Increasing by the pervious image's max positional value
        data['X'] = data['X'] + max_value

        # Append the data to the combined data frame
        data_df = pd.concat([data_df, data], ignore_index=True)

    # Negative Y-axis 
    elif experimental_csv.find("T")  != -1: # Bellow the crator
        # Create a new DataFrame for the data
        data = pd.DataFrame(data=experimental_data, columns=["X", "Y", "Width"])
        
        # Set Values to zero. 
        data["Y"] *= 0
        
        # Increasing the distance from the creator for further images
        max_value = data["X"].max() # max value of the current image
        min_value = data_df["X"].min() # min value of the whole dataset


        # Increasing by the pervious image's max positional value
        data['X'] = min_value - (max_value - data['X'])

        # Append the data to the combined data frame
        data_df = pd.concat([data_df, data], ignore_index=True)

    else:
        print(f"this file failed to be adjusted {experimental_csv}")

    return data_df


def load_experimental_data(experiment_type : str):
    """
    Load experimental data for a specific experiment mode.

    Args:
        experiment_type (str): The type of experiment mode (no B-field, B into page, B out of page).

    Returns:
        pd.DataFrame: A DataFrame containing the loaded experimental data.
    """

    # Create the experimental data frame - use Zeros to initialize the DF with values (removed later)
    data_df = pd.DataFrame({
        "X": [0],
        "Y": [0],
        "Width": [0]
    })

    # Gets the file names for the folder
    experimental_csv_list = get_file_names("data/" + experiment_type)

    # Loads the data from each file in the folder
    for experimental_csv in experimental_csv_list:
        
        # loads the data from the CSV file
        experimental_data = load_data_from_csv("data/" + experiment_type + "/" + experimental_csv)

        # Based on where the data was recorded relative to the creator, values need to be adjusted
        if experiment_type.find("Across") != -1:
            # The Bfield across the page has a strange oreintation, and needs to be correct due B field direction
            data_df = Experimental_special_adjustment(experimental_csv, experimental_data, data_df)
        else:
            data_df = experimental_adjustment(experimental_csv, experimental_data, data_df)

    # Remove the zeros used to initialize the DataFrame
    data_df = data_df.iloc[1:]

    return data_df

# Function to write data to a JSON file
def write_to_json(data, filename : str='output1.json'):
    data_serializable = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data_serializable[key] = value.tolist()
        else:
            data_serializable[key] = value
    with open(filename, 'w') as json_file:
        json.dump(data_serializable, json_file)

# loads data from a JSON File
def load_from_json(filename : str ='output1.json'):
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return None


def save_dataframe(df: pd.DataFrame, file_path: str, force_format: str = None, headers=True, index = False, silent_mode = False):
    """
    Save a pandas DataFrame to a specified file format.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - file_path (str): The path where the DataFrame should be saved.
    - force_format (str, optional): Desired file format to force. Supported formats: 'parquet' (recommended), 'csv', 'excel', 'txt'.

    Returns:
    - bool: True if saving was successful, False otherwise.
    """
    try:
        # Determine the file format to use
        file_format = force_format or file_path.split('.')[-1].lower()
        
        if file_format == 'parquet': # the binary type --> fastest
            df.to_parquet(file_path)
        elif file_format == 'csv':
            df.to_csv(file_path, index=index)
        elif file_format == 'excel' or file_format == 'xlsx':
            df.to_excel(file_path, index=index)
        elif file_format == 'txt':
            df.to_csv(file_path, sep='\t', index=index, header=headers)
        elif file_format == "json":
            write_to_json(df, file_path)
  
        else:
            # Incase someone wants to save to an unsupported formate
            raise ValueError(f"Unsupported file format: {file_format}")

        if not silent_mode:
            print(f"DataFrame successfully saved to {file_path}")

        return True

    except Exception as e:
        print(f"Error saving DataFrame: {e}")
        return False
    
def load_dataframe(file_path: str, enable_pyarrow = False):
    """
    Load a pandas DataFrame from a specified file format.
    
    Parameters:
    - file_path (str): The path to the file to load.
    - enable_pyarrow (bool): Eanbled an optimised method for loading parquet files

    Returns:
    - pd.DataFrame: The loaded DataFrame if successful, None otherwise.
    """
    try:
        # gets the file extension
        file_format = file_path.split('.')[-1].lower()

        
        if file_format == 'parquet': # the binary type --> fastest
            if enable_pyarrow:
                # This may cause loading problems for strange DataFrames
                df = pd.read_parquet(file_path, engine='pyarrow') # Use the C++ optimised engine for loading 
            else:
                df = pd.read_parquet(file_path) # let pandas decide what engine to use
        elif file_format == 'csv':
            load_data_from_csv(file_path)
        elif file_format in ['xls', 'xlsx']: # For the various types of Excel file types
            load_data_from_excel(file_path)
        elif file_format == 'txt':
            load_data_from_txt(file_path)
        elif file_format == "json":
            load_from_json(file_path)
        else:
            # For the losers who want to load un-listed formats
            raise ValueError(f"Unsupported file format: {file_format}")

        # Check if the DataFrame is empty
        if df.empty:
            print(f"Warning: The DataFrame loaded from {file_path} is empty.")
            return None

        print(f"DataFrame successfully loaded from {file_path}")
        return df

    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        return None
    

def bulk_dump_columns(data :  pd.DataFrame, basepath : str, file_fromatte : str = '.txt', field : str = 'Wavelength (nm)', excluded_fields : list = []):  
    """
    Exports multiple columns from a DataFrame to individual text files, excluding specified fields.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to be exported.
        basepath (str): The base directory path where the files will be saved.
        file_format (str): The file extension format for the output files (default: '.txt').
        field (str): The column name to use as a reference or index (default: 'Wavelength (nm)').
        excluded_fields (list): A list of column names to exclude from the export (default: []).

    Returns:
        None

    """

    if field not in data.columns:
        print(f"Warning: {field} field can not be found in prodived data. Please select from {data.columns}")
        return None

    # Adds the wavelength field to the excluded fields
    excluded_fields.append(field)

    data_columns = data.drop(columns=excluded_fields, axis=1).columns.tolist()

    for column in data_columns:
        # Gets the completed path name
        save_path = basepath + column + file_fromatte

        # Creates a single data frame with the wavelength and series of data
        df = data[[field, column]]

        # Saves the dataframe to the spcificed path
        save_dataframe(df, save_path, headers=False, silent_mode=True)

    print(f'Saved {len(data_columns)} files')


def load_multiple_outputs(basepath : str, field : str, file_filter : str = ".txt", col_formatter : str = ".") -> pd.DataFrame:
    """
    Loads and combines multiple text files from a specified directory into a single pandas DataFrame.
    
    Each text file should contain two columns: one for particle radii and another for particle concentration.
    The function merges these data files based on the particle size (radii) and uses the filenames as column headers
    for the concentration data.

    Args:
        basepath (str): The base directory path where the text files are located.
        field (str): The name of the first column representing the particle radii.
        file_filter (str, optional): The file extension filter to select files from the base directory (default is ".txt").
        col_formatter (str, optional): The delimiter used to format column names from filenames (default is ".").
                                       Only the part of the filename before the first occurrence of this delimiter
                                       will be used as the column name.

    Returns:
        pd.DataFrame: A merged DataFrame containing all the data from the text files, where the first column is the
                      particle size (radii) and subsequent columns represent the concentration data from each file.

    Raises:
        FileNotFoundError: If no files matching the filter are found in the base directory.
        ValueError: If any file cannot be loaded into a DataFrame or if merging fails.
    
    Example:
        >>> load_df_from_txts('/data/particles/', 'Radii', '.txt', '_norm')
        Returns a DataFrame where the first column is 'Radii' and subsequent columns are concentrations for different 
        files, named after their respective file prefixes.
    """
    # Retive the files names from base path that match the file_filter patten
    file_names = get_file_names(basepath, file_filter)

    # This list will hold the df for each of the files loaded
    df_list = []

    # Generate a df for each file in the basepath
    # and add it to the df_list --> will merge latter
    for name in file_names:

        if file_filter == '.csv':
            df = load_data_from_csv(basepath + name)
        else:
            df = load_data_from_txt(basepath + name, header = None)


        # Chnaging the column names
        # col_formatter will cut the stings off and take the first half
        # i.e "1.8W - 21-08 - 14Ks_norm.txt" col_formatter = "." --> "1.8W - 21-08 - 14Ks_norm"
        df.columns = [field,  name.split(col_formatter)[-1]] 

        # Check if the fild is on the nano-scale with SI units
        # if so, convert to nm from SI units
        # Else assume alread in nm units
        if df[field].max() <= 1e-6:
            df[field] *= 1e9

        # Adding the df to the list
        df_list.append(df)

    # need an initial df to merge the others onto
    merged_df = df_list[0]
    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, on=field, how='outer')  # Outer join to ensure all radii are included

    return merged_df

