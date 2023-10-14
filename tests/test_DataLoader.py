import pytest
import pandas as pd
from src.DataLoader import *

# Define test data file paths
existing_csv_path = 'data/BFieldIn/Left1.csv'
non_existing_csv_path = 'path_to_non_existing_file.csv'

# Create a fixture to load data
@pytest.fixture
def existing_data():
    data = load_data_from_csv(existing_csv_path)
    yield data
    data = None  # Clean up

@pytest.fixture
def non_existing_data():
    data = load_data_from_csv(non_existing_csv_path)
    yield data
    data = None  # Clean up

@pytest.fixture
def sample_files(tmpdir):
    # Define a list of sample filenames
    filenames = ['file1.txt', 'file2.csv', 'file3.xlsx']

    # Create the sample files in the temporary directory
    for filename in filenames:
        tmpdir.join(filename).ensure(file=True)

    yield tmpdir

def test_load_data_from_csv_existing(existing_data):
    assert isinstance(existing_data, pd.DataFrame)
    # Add additional assertions for the loaded data if needed

def test_load_data_from_csv_non_existing(non_existing_data):
    assert non_existing_data is None


def test_get_file_names(sample_files):
    folder_path = str(sample_files)

    # Call the function to get file names
    filenames = get_file_names(folder_path)

    # Check if the filenames match the expected list
    expected_filenames = ['file1.txt', 'file2.csv', 'file3.xlsx']
    assert set(filenames) == set(expected_filenames)


def test_experimental_adjustment():
    # Example data for testing
    experimental_csv = "R_Exp.csv"
    experimental_data = pd.DataFrame({"X": [1, 2, 3], "Y": [4, 5, 6], "Width": [7, 8, 9]})
    data_df = pd.DataFrame({"X": [0], "Y": [0], "Width": [0]})

    # Call the function to adjust and append data
    updated_data_df = experimental_adjustment(experimental_csv, experimental_data, data_df)

    # Check the resulting data frame
    expected_data = pd.DataFrame({
        "X": [0, 1, 2, 3],
        "Y": [0, 0, 0, 0],
        "Width": [0, 7, 8, 9]
    })

    # Ensure the data frames match
    pd.testing.assert_frame_equal(updated_data_df, expected_data)