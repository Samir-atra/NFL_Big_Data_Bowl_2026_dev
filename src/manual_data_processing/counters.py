"""
Script for running exploratory data analysis (EDA) to count and check frame
consistency in the NFL Big Data Bowl datasets.

This module provides utilities to:
1. Count the number of time frames for each unique (game_id, play_id, nfl_id)
   combination in both input and output files, specifically focusing on the
   'player_to_predict' in the input data.
2. Generate descriptive statistics for frame counts to check for sequence length
   uniformity.
3. Verify the correspondence between unique player-play identifiers in the input
   and output datasets to ensure a complete training set.
"""
import pandas as pd
import os
import glob

TRAIN_DIR = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'

def count_frames(file_pattern, label):
    """
    Scans CSV files matching a pattern and counts the number of frames (rows)
    for each unique play-player combination.

    For 'INPUT' files, it filters for the 'player_to_predict'.

    Args:
        file_pattern (str): Glob pattern for the files to scan (e.g., 'input_*.csv').
        label (str): Descriptive label for the output (e.g., 'INPUT' or 'OUTPUT').
    """
    print(f"\n--- Scanning {label} files in {TRAIN_DIR} ---")
    
    files = sorted(glob.glob(os.path.join(TRAIN_DIR, file_pattern)))
    
    if not files:
        print(f"No {label} files found.")
        return

    all_counts = []

    for file_path in files:
        # Read only necessary ID columns to save memory and process only relevant data
        try:
            # Peek at columns to determine which ones are available
            peek_df = pd.read_csv(file_path, nrows=0)
            cols_to_use = ['game_id', 'play_id', 'nfl_id']
            
            # Conditionally add 'player_to_predict' for input files
            if label == 'INPUT' and 'player_to_predict' in peek_df.columns:
                cols_to_use.append('player_to_predict')

            available_cols = [c for c in cols_to_use if c in peek_df.columns]
            
            if not available_cols:
                 print(f"Skipping {os.path.basename(file_path)}: No ID columns found.")
                 continue

            df = pd.read_csv(file_path, usecols=available_cols)
            
            # Filter for the player whose future movement we need to predict
            if label == 'INPUT' and 'player_to_predict' in df.columns:
                # Convert to boolean/string and filter for 'True'
                df = df[df['player_to_predict'].astype(str).str.lower() == 'true']
                
                # Exclude 'player_to_predict' from the grouping columns
                group_cols = [c for c in available_cols if c != 'player_to_predict']
            else:
                group_cols = available_cols
            
            # Group by available IDs (e.g., game_id, play_id, nfl_id) and count rows/frames
            counts = df.groupby(group_cols).size().reset_index(name='frame_count')
            
            all_counts.append(counts)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not all_counts:
        print(f"No data processed for {label}.")
        return

    # Aggregate and summarize counts across all weeks
    final_df = pd.concat(all_counts, ignore_index=True)
    
    print(f"\n--- {label} Frame Counts Summary ---")
    print(final_df['frame_count'].describe())
    
    unique_counts = sorted(final_df['frame_count'].unique())
    print(f"\nUnique frame counts found: {unique_counts[:20]} ... {unique_counts[-5:] if len(unique_counts) > 20 else ''}")
    
    # Check for consistency
    if len(unique_counts) == 1:
        print(f"CONFIRMED: All {label} groups have exactly {unique_counts[0]} frames.")
    else:
        print(f"WARNING: Variable number of frames detected in {label}. This may require padding.")

def check_id_correspondence():
    """
    Checks if all unique player-play combinations present in the input files
    (filtered by 'player_to_predict') also exist in the output files.
    This verifies the completeness of the dataset for a supervised task.
    """
    print("\n--- Checking ID Correspondence (Game, Play, NFL ID) ---")
    
    input_files = sorted(glob.glob(os.path.join(TRAIN_DIR, 'input_*.csv')))
    output_files = sorted(glob.glob(os.path.join(TRAIN_DIR, 'output_*.csv')))
    
    if not input_files or not output_files:
        print("Missing input or output files.")
        return

    # Use sets for efficient storage and intersection checking of unique identifiers: (game_id, play_id, nfl_id)
    input_ids = set()
    output_ids = set()

    print("Reading Input IDs...")
    for f in input_files:
        try:
            # Prepare to read ID columns and the prediction flag
            peek = pd.read_csv(f, nrows=0)
            cols_to_use = ['game_id', 'play_id', 'nfl_id']
            if 'player_to_predict' in peek.columns:
                cols_to_use.append('player_to_predict')
            
            df = pd.read_csv(f, usecols=cols_to_use)
            
            # Filter to only include the player we are interested in predicting
            if 'player_to_predict' in df.columns:
                df = df[df['player_to_predict'].astype(str).str.lower() == 'true']
            
            # Collect unique (game_id, play_id, nfl_id) tuples
            unique_tuples = set(df[['game_id', 'play_id', 'nfl_id']].itertuples(index=False, name=None))
            input_ids.update(unique_tuples)
        except Exception as e:
            print(f"Error reading {os.path.basename(f)}: {e}")

    print("Reading Output IDs...")
    for f in output_files:
        try:
            # Check for required ID columns in output
            peek = pd.read_csv(f, nrows=0)
            cols = ['game_id', 'play_id', 'nfl_id']
            available = [c for c in cols if c in peek.columns]
            
            if len(available) < 3:
                # Skip if the necessary identifiers for matching are not present
                continue

            df = pd.read_csv(f, usecols=available)
            # Collect unique (game_id, play_id, nfl_id) tuples
            unique_tuples = set(df[available].itertuples(index=False, name=None))
            output_ids.update(unique_tuples)
        except Exception as e:
            print(f"Error reading {os.path.basename(f)}: {e}")

    print(f"\nTotal Unique Input Combinations (Player-Play): {len(input_ids)}")
    print(f"Total Unique Output Combinations (Player-Play): {len(output_ids)}")
    
    # Calculate Intersection: Plays that exist in both input and output
    common_ids = input_ids.intersection(output_ids)
    print(f"Matching Combinations (Input has Output): {len(common_ids)}")
    
    if len(input_ids) > 0:
        # Report coverage percentage
        match_percent = (len(common_ids) / len(input_ids)) * 100
        print(f"Percentage of Input covered by Output: {match_percent:.2f}%")
    
    missing_in_output = len(input_ids) - len(common_ids)
    print(f"Input Combinations missing in Output: {missing_in_output}")

if __name__ == "__main__":
    count_frames('input_*.csv', 'INPUT')
    count_frames('output_*.csv', 'OUTPUT')
    check_id_correspondence()
