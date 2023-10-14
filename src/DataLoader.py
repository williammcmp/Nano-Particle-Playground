# src/DataLoader.py

import pandas as pd
import os

def load_data_from_excel(file_path, sheet_name):
    """
    Load data from an Excel file.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str): The name of the Excel sheet to read.

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing the data from the Excel file.
            Returns None if the file is not found or an error occurs during loading.
    """
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        return data
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from Excel file: {file_path}")
        return None

def load_data_from_csv(file_path):
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

def get_file_names(folder_path):
    """
    Get a list of filenames in the specified folder path.

    Args:
        folder_path (str): The path to the folder from which to retrieve filenames.

    Returns:
        list of str: A list of filenames in the specified folder.
    """
    return os.listdir(folder_path)

# Adjust values depending on what side of the crator they are recorded from
def experimental_adjustment(experimental_csv, experimental_data, data_df):
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


def load_experimental_data(experiment_type):
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
        data_df = experimental_adjustment(experimental_csv, experimental_data, data_df)

    # Remove the zeros used to initialize the DataFrame
    data_df = data_df.iloc[1:]

    return data_df
