"""
Manual data loading utility using the Pandas library.

This module provides a simple, standalone function `load_dataframe` designed
to load a CSV file into a Pandas DataFrame, ensuring basic file existence
checks and providing informational output on the loaded data size.
"""
import pandas as pd
import os

def load_dataframe(file_path):
    """
    Loads a CSV file into a Pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    print(f"Loading data from {file_path}...")
    # Low_memory=False to handle potential mixed types across columns
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded {len(df)} rows.")
    return df
