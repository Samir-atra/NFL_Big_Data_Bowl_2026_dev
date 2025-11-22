import pandas as pd
import os
import glob

TRAIN_DIR = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'

def count_frames(file_pattern, label):
    print(f"\n--- Scanning {label} files in {TRAIN_DIR} ---")
    
    files = sorted(glob.glob(os.path.join(TRAIN_DIR, file_pattern)))
    
    if not files:
        print(f"No {label} files found.")
        return

    all_counts = []

    for file_path in files:
        # print(f"Processing {os.path.basename(file_path)}...")
        try:
            # Read only necessary columns to save memory
            # Note: Output files might not have 'nfl_id' if they are just labels per play, 
            # but usually they map to the input structure. We'll check columns first.
            
            # Peek at columns
            peek_df = pd.read_csv(file_path, nrows=0)
            cols_to_use = ['game_id', 'play_id', 'nfl_id']
            
            # Add 'player_to_predict' if we are processing INPUT files and it exists
            if label == 'INPUT' and 'player_to_predict' in peek_df.columns:
                cols_to_use.append('player_to_predict')

            available_cols = [c for c in cols_to_use if c in peek_df.columns]
            
            if not available_cols:
                 print(f"Skipping {os.path.basename(file_path)}: No ID columns found.")
                 continue

            df = pd.read_csv(file_path, usecols=available_cols)
            
            # Filter for player_to_predict == True (handling boolean or string 'True')
            if label == 'INPUT' and 'player_to_predict' in df.columns:
                # Convert to boolean just in case it's read as string/object
                # Assuming 'True' string or True boolean
                df = df[df['player_to_predict'].astype(str).str.lower() == 'true']
                
                # We don't need to group by 'player_to_predict' anymore since it's all True
                group_cols = [c for c in available_cols if c != 'player_to_predict']
            else:
                group_cols = available_cols
            
            # Group by available IDs and count rows
            counts = df.groupby(group_cols).size().reset_index(name='frame_count')
            
            all_counts.append(counts)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not all_counts:
        print(f"No data processed for {label}.")
        return

    final_df = pd.concat(all_counts, ignore_index=True)
    
    print(f"\n--- {label} Frame Counts Summary ---")
    print(final_df['frame_count'].describe())
    
    unique_counts = sorted(final_df['frame_count'].unique())
    print(f"\nUnique frame counts found: {unique_counts[:20]} ... {unique_counts[-5:] if len(unique_counts) > 20 else ''}")
    
    if len(unique_counts) == 1:
        print(f"CONFIRMED: All {label} groups have exactly {unique_counts[0]} frames.")
    else:
        print(f"WARNING: Variable number of frames detected in {label}.")

def check_id_correspondence():
    print("\n--- Checking ID Correspondence (Game, Play, NFL ID) ---")
    
    input_files = sorted(glob.glob(os.path.join(TRAIN_DIR, 'input_*.csv')))
    output_files = sorted(glob.glob(os.path.join(TRAIN_DIR, 'output_*.csv')))
    
    if not input_files or not output_files:
        print("Missing input or output files.")
        return

    # Use sets to store unique identifiers: (game_id, play_id, nfl_id)
    input_ids = set()
    output_ids = set()

    print("Reading Input IDs...")
    for f in input_files:
        try:
            # Read ID columns AND player_to_predict
            # We need to filter by player_to_predict == True
            peek = pd.read_csv(f, nrows=0)
            cols_to_use = ['game_id', 'play_id', 'nfl_id']
            if 'player_to_predict' in peek.columns:
                cols_to_use.append('player_to_predict')
            
            df = pd.read_csv(f, usecols=cols_to_use)
            
            # Filter for player_to_predict == True
            if 'player_to_predict' in df.columns:
                df = df[df['player_to_predict'].astype(str).str.lower() == 'true']
            
            # Add unique tuples to the set
            unique_tuples = set(df[['game_id', 'play_id', 'nfl_id']].itertuples(index=False, name=None))
            input_ids.update(unique_tuples)
        except Exception as e:
            print(f"Error reading {os.path.basename(f)}: {e}")

    print("Reading Output IDs...")
    for f in output_files:
        try:
            # Output files might not have nfl_id if they are play-level, but assuming they do based on previous context
            # We check columns first
            peek = pd.read_csv(f, nrows=0)
            cols = ['game_id', 'play_id', 'nfl_id']
            available = [c for c in cols if c in peek.columns]
            
            if len(available) < 3:
                # If nfl_id is missing, we can't match on it. 
                # But let's assume for now we want to match on whatever is available or strictly all 3.
                # If nfl_id is missing in output, we can't strictly answer "how many input rows have output with equal nfl id".
                # We will skip if nfl_id is missing to be strict.
                continue

            df = pd.read_csv(f, usecols=available)
            unique_tuples = set(df[available].itertuples(index=False, name=None))
            output_ids.update(unique_tuples)
        except Exception as e:
            print(f"Error reading {os.path.basename(f)}: {e}")

    print(f"\nTotal Unique Input Combinations: {len(input_ids)}")
    print(f"Total Unique Output Combinations: {len(output_ids)}")
    
    # Calculate Intersection
    common_ids = input_ids.intersection(output_ids)
    print(f"Matching Combinations (Input has Output): {len(common_ids)}")
    
    if len(input_ids) > 0:
        match_percent = (len(common_ids) / len(input_ids)) * 100
        print(f"Percentage of Input covered by Output: {match_percent:.2f}%")
    
    missing_in_output = len(input_ids) - len(common_ids)
    print(f"Input Combinations missing in Output: {missing_in_output}")

if __name__ == "__main__":
    count_frames('input_*.csv', 'INPUT')
    count_frames('output_*.csv', 'OUTPUT')
    check_id_correspondence()
