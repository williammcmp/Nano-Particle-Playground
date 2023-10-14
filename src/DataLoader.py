# src/DataLoader.py

import pandas as pd
import os

# Loads data from excel file
def load_data_from_excel(file_path, sheet_name):
    try:
        # Load data from the Excel file
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        return data
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from Excel file: {file_path}")
        return None  # Return None if there's an error
    
def load_data_from_csv(file_path):
    try:
        # Read data from the CSV file into a pandas DataFrame
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from CSV file:: {str(e)}")
        return None
    
#  Gets a list of files for the specified folder path
def get_file_names(folder_path):
    return os.listdir(folder_path)

# Adjust values depending on what side of the crator they are recorded from
def experimental_adjustment(experimental_csv, experimental_data, data_df):
    
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


#  Loads data from a specific experiment mode (no B-filed, B Into page, B Out of page)
def load_experimental_data(experiment_type):
    # Create the experimental data frame - use Zeros to initalise the DF with values (removed latter)
    data_df = pd.DataFrame({
        "X": [0],
        "Y": [0],
        "Width": [0]
    })

    # Gets the file names for the folder 
    experimental_csv_list = get_file_names("data/"+experiment_type)

    # Loads the data from each file in the folder
    for experimental_csv in experimental_csv_list:
        
        # loads the data from excel file
        experimental_data = load_data_from_csv("data/"+experiment_type+"/"+experimental_csv)

        # based on where the data was recorded relative to the creator, values need to be adjusted
        data_df = experimental_adjustment(experimental_csv, experimental_data, data_df)

    # Remove the zeros used to initalis the DataFrame 
    data_df = data_df.iloc[1:]

    return data_df





data_df = load_experimental_data("BFieldIn")


import matplotlib.pyplot as plt

# Assuming you already have your DataFrame data_df

# Plot the 'X' and 'Y' columns
plt.scatter(data_df['X'], data_df['Y'])

# Customize the plot (optional)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs. Y')

# Show the plot
plt.show()
