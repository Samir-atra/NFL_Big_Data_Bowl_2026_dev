
import pandas as pd
import numpy as np

try:
    file_path = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/cached_data/y_val.parquet'
    df = pd.read_parquet(file_path)
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst row sample:")
    print(df.iloc[0])
    
    # Check if it's a flattened array or structured data
    # The previous context suggested it might be a numpy array saved as parquet, 
    # so columns might just be '0', '1', '2' etc. or it might have feature names.
except Exception as e:
    print(e)
