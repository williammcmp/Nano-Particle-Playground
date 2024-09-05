# src/DataLoader.py

import pandas as pd
import os
import numpy as np
import json

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


def normalize_data(data : pd.DataFrame, exclude : list = ['Radii(nm)'], mode = 'min-max') -> pd.DataFrame:
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
    # Stops repeated column normalistions
    columns = data.columns.copy()

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

def load_mathematica_outputs(basepath : str, field : str, file_filter : str = ".txt", col_formatter : str = ".") -> pd.DataFrame:
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
        df = load_data_from_txt(basepath + name, header = None)

        # Chnaging the column names
        # col_formatter will cut the stings off and take the first half
        # i.e "1.8W - 21-08 - 14Ks_norm.txt" col_formatter = "." --> "1.8W - 21-08 - 14Ks_norm"
        df.columns = [field,  name.split(col_formatter)[0]] 

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
