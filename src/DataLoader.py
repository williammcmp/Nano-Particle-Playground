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
    
def load_data_from_txt(file_path : str):
    """
    Load data from a .txt file.

    Args:
        file_path (str): The path to the .txt file.

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing the data from the .txt file.
            Returns None if the file is not found or an error occurs during loading.
    """
    try:
        data = pd.read_csv(file_path, sep='\t')
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

def get_file_names(folder_path : str):
    """
    Get a list of filenames in the specified folder path.

    Args:
        folder_path (str): The path to the folder from which to retrieve filenames.

    Returns:
        list of str: A list of filenames in the specified folder.
    """
    return os.listdir(folder_path)

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


def normalize_data(data : pd.DataFrame, exclude : list = ['Radii(nm)']):
    """
    Normalize the data in all columns except those in the exclude list and add new columns with normalized data.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to normalize.
        exclude (list): List of column names to exclude from normalization.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for normalized data.
    """
    # Stops repeated column normalistions
    columns = data.columns.copy()

    for column in columns:
        # Checks if the column is not in the exclude list
        # Skips excluded columns
        if column not in exclude:
            max_value = data[column].max()
            min_value = data[column].min()
            data[column + '_norm'] = (data[column] - min_value) / (max_value - min_value)

    return data  

def remove_offset(data: pd.DataFrame, wavelength: int = 300):
    """
    Removes the offset from a given DataFrame based on a specified wavelength.

    This function checks if the specified wavelength is within the range of the DataFrame's 'Wavelength (nm)' column.
    If the wavelength is out of range, a warning is printed. The function then calculates the average differential
    offset for wavelengths below the specified value and adjusts the DataFrame values accordingly.

    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame containing at least one column named 'Wavelength (nm)' and other columns to adjust.

    wavelength : int, optional
        The reference wavelength (in nm) to use for offset adjustment. Default is 300 nm.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame with the offset removed for values below the specified wavelength.

    Notes:
    -----
    - The function assumes that the DataFrame has a column named 'Wavelength (nm)' and that it is numeric.
    - The operation assumes a continuous range of wavelengths; missing values or non-numeric data may cause errors.
    - If the specified wavelength is not within the 'Wavelength (nm)' column, the function issues a warning and terminates.
    
    Raises:
    ------
    KeyError
        If 'Wavelength (nm)' is not a column in the input DataFrame.
    
    IndexError
        If no valid indices are found for the given masks.

    Example:
    --------
    >>> df = pd.DataFrame({'Wavelength (nm)': [290, 295, 300, 305], 'Intensity': [1.2, 1.5, 1.8, 2.1]})
    >>> corrected_df = remove_offset(df, wavelength=300)
    """

    if not (data['Wavelength (nm)'] == wavelength).any():
        print(f'Warning: Wavelength {wavelength}nm is out of range. Please select a wavelength that is valid to the data set.')
        return

    lower_mask = (data['Wavelength (nm)'] < wavelength - 10) & (data['Wavelength (nm)'] > wavelength - 10)

    center_mask = data['Wavelength (nm)'].isin([wavelength - 10, wavelength])

    lower_diff = np.diff(data[lower_mask], axis=0)
    center_diff = np.diff(data[center_mask], axis=0)

    lower_diff = np.average(lower_diff, axis=0)

    lower_offset = center_diff - lower_diff

    move_mask = data['Wavelength (nm)'] < wavelength

    corrected = data.copy()
    corrected[move_mask] = corrected[move_mask] - lower_offset

    print(corrected['Wavelength (nm)'] < 300)

    return corrected

def save_dataframe(df: pd.DataFrame, file_path: str, force_format: str = None, headers=True, index = False):
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
    

def bulk_dump_columns(data :  pd.DataFrame, basepath : str, file_fromatte : str = '.txt'):  
    # Will save all the columns to seperate files with the wavelength for the index column
    data_columns =  data.columns[1:]  # Assuming first column is 'Wavelength (nm)'
    for column in data_columns:
        save_path = basepath + column + file_fromatte
        df = data[['Wavelength (nm)', column]]
        save_dataframe(df, save_path, headers=False)

    print(f'{len(data_columns)} files have been saved.')