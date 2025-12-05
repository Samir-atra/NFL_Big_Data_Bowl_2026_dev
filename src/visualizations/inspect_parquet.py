
"""
Utility script to inspect the content and structure of a Parquet file.

This tool is specifically used for debugging and verifying the integrity and format
of cached data files (like those saved in the 'cached_data' directory) by loading
the file, printing its dimensions (shape), column names, and a sample of the first row.
"""
import pandas as pd
import numpy as np

# Define the path to the Parquet file intended for inspection.
try:
    file_path = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/cached_data/y_val.parquet'
    
    # Load the Parquet file into a Pandas DataFrame
    df = pd.read_parquet(file_path)
    
    # Print basic information for verification
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst row sample:")
    print(df.iloc[0])
    
    # Check if it's a flattened array or structured data
    # The previous context suggested it might be a numpy array saved as parquet, 
    # so columns might just be '0', '1', '2' etc. or it might have feature names.
except Exception as e:
    # Print the error if the file cannot be read or found
    print(f"Error inspecting Parquet file: {e}")
