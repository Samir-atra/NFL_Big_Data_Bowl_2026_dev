"""
Module for calculating statistical properties of the NFL Big Data Bowl training data.

This script uses the NFLDataLoader to load and align the sequence data (input features X
and output targets y) and then computes the global mean and standard deviation
for both X and y across all time steps in the dataset. These statistics are crucial
for standardizing the data during model training and inference. The computed
statistics are typically saved to a pickle file for later use.
"""
import numpy as np
import os
import sys
import pickle

# Add the current directory to path to allow imports if run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from csv_to_numpy import NFLDataLoader

def calculate_stats(train_dir, save_path=None):
    """
    Calculates the mean and standard deviation of the input features (X)
    and output targets (y) loaded by NFLDataLoader.
    
    Args:
        train_dir (str): Path to the training directory.
        save_path (str, optional): Path to save the statistics (as a pickle or npz).
        
    Returns:
        dict: A dictionary containing 'X_mean', 'X_std', 'y_mean', 'y_std'.
    """
    loader = NFLDataLoader(train_dir)
    X, y = loader.get_aligned_data()
    
    if len(X) == 0:
        print("No data found to calculate stats.")
        return None

    print("Data loaded. Calculating statistics...")
    
    # --- Process X (Inputs) ---
    print("Processing Input (X) data...")
    # X is an array of sequences. Each sequence is a list of steps.
    # We flatten all sequences to shape (total_steps, n_features)
    
    all_X_steps = []
    for seq in X:
        all_X_steps.extend(seq)
        
    X_array = np.array(all_X_steps)
    print(f"Total input time steps: {X_array.shape[0]}")
    print(f"Input features: {X_array.shape[1]}")
    
    X_mean = np.mean(X_array)
    X_std = np.std(X_array)
    
    if X_std == 0:
        X_std = 1.0
    
    # --- Process y (Outputs) ---
    print("Processing Output (y) data...")
    all_y_steps = []
    for seq in y:
        all_y_steps.extend(seq)
        
    y_array = np.array(all_y_steps)
    print(f"Total output time steps: {y_array.shape[0]}")
    print(f"Output features: {y_array.shape[1]}")
    
    y_mean = np.mean(y_array)
    y_std = np.std(y_array)
    
    if y_std == 0:
        y_std = 1.0

    stats = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'n_input_features': X_array.shape[1],
        'n_output_features': y_array.shape[1]
    }
    
    print("\nStatistics calculated.")
    print(f"X Mean: {X_mean}")
    print(f"X Std: {X_std}")
    
    if save_path:
        print(f"Saving statistics to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(stats, f)
        print("Saved.")
        
    return stats

if __name__ == "__main__":
    # Default path, can be modified or passed as arg
    TRAIN_DIR = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    SAVE_FILE = os.path.join(os.path.dirname(TRAIN_DIR), 'data_stats.pkl')
    
    calculate_stats(TRAIN_DIR, save_path=SAVE_FILE)
