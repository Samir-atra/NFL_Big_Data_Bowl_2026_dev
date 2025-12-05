"""
This module provides a utility to inspect CSV files from the NFL Big Data Bowl 2026 dataset.
It allows users to specify a filename via the command line, and the script will then:

1. Locate the file either in the current directory or a predefined base directory.
2. Read the CSV file using pandas.
3. Print key information about the file, including:
    - Column names
    - Shape (number of rows and columns)
    - Data types of each column
    - The first 5 rows of the data

This tool is useful for quickly understanding the structure and content of dataset files.
"""

import pandas as pd
import argparse
import os

# Define the base directory where prediction training data is stored.
# This is used as a fallback location if the file isn't found in the current directory.
BASE_DIR = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train'

def inspect_file(filename: str) -> None:
    """
    Inspects a specified CSV file by reading it into a pandas DataFrame
    and printing its columns, shape, data types, and the first few rows.

    The function first checks for the file in the current directory, and if
    not found, it looks in the predefined BASE_DIR.

    Args:
        filename (str): The name of the CSV file to inspect.
    """
    # Determine the full path to the file.
    # Check if the file exists in the current working directory.
    if os.path.exists(filename):
        filepath = filename
    # If not found, check if it exists in the BASE_DIR.
    elif os.path.exists(os.path.join(BASE_DIR, filename)):
        filepath = os.path.join(BASE_DIR, filename)
    else:
        # If the file is not found in either location, print an error message and return.
        print(f"Error: File '{filename}' not found in current directory or {BASE_DIR}")
        return

    print(f"--- Inspecting: {filepath} ---")
    try:
        # Read the CSV file into a pandas DataFrame.
        df = pd.read_csv(filepath)
    except Exception as e:
        # Handle potential errors during file reading (e.g., corrupted file, permissions).
        print(f"Error reading file: {e}")
        return

    # Print key information about the DataFrame.
    print("\n[Columns]")
    print(df.columns.tolist()) # List of all column names

    print("\n[Shape]")
    print(df.shape) # Tuple representing (number of rows, number of columns)

    print("\n[Types]")
    print(df.dtypes) # Data types of each column

    print("\n[First 5 Lines]")
    print(df.head()) # Display the first 5 rows of the DataFrame

# This block executes only when the script is run directly (not imported as a module).
if __name__ == "__main__":
    # Set up argument parsing to accept the filename from the command line.
    parser = argparse.ArgumentParser(description="Inspect a CSV file.")
    # Add a required positional argument for the filename.
    parser.add_argument("filename", help="Name of the file to inspect")
    
    # Parse the command-line arguments.
    args = parser.parse_args()

    # Call the inspect_file function with the provided filename.
    inspect_file(args.filename)
