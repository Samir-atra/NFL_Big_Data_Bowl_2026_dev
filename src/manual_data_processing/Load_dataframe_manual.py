import pandas as pd
import os

def load_dataframe(file_path):
    """
    Loads a CSV file into a Pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    print(f"Loading data from {file_path}...")
    # Low_memory=False to handle mixed types if necessary, though for this dataset standard read is usually fine
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded {len(df)} rows.")
    return df
