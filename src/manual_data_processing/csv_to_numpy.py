"""
Module for converting raw NFL Big Data Bowl CSV data into aligned NumPy sequence arrays.

This module defines the NFLDataLoader class which uses the Polars library for
efficient loading, feature processing (including handling of categorical/string
data and date conversion), and alignment of input (X) and output (y) time series data.
It returns NumPy object arrays of variable-length sequences, which are then typically
used with the NFLDataSequence (a Keras Sequence) to manage on-the-fly padding and batching.
"""
import polars as pl
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence

class NFLDataLoader:
    """Loads and processes NFL Big Data Bowl 2026 data from CSV files using Polars.

    This class handles the loading of input and output CSV files, filtering for
    specific players, and aligning input sequences with their corresponding
    output sequences based on game, play, and NFL IDs.

    Attributes:
        train_dir (str): The directory containing the training CSV files.
        input_sequences (pl.DataFrame): DataFrame containing input sequences (grouped lists of features).
        output_sequences (pl.DataFrame): DataFrame containing output sequences (grouped lists of targets).
    """
    def __init__(self, train_dir):
        """
        Initializes the DataLoader with the training data directory.

        Args:
            train_dir (str): Path to the directory containing input and output CSV files.
        """
        self.train_dir = train_dir
        self.input_sequences = None
        self.output_sequences = None

    def load_input_files(self):
        """
        Loads and filters input CSV files from the training directory using Polars.

        It processes all input files, filters rows where 'player_to_predict' is True,
        applies extensive vectorized feature processing to convert strings and
        categorical data to floats, and finally groups the processed features by
        (game_id, play_id, nfl_id) to form sequences of feature lists.
        """
        input_files = sorted([f for f in os.listdir(self.train_dir) if f.startswith('input') and f.endswith('.csv')])
        print(f"Loading and filtering {len(input_files)} Input files...")
        
        dataframes = []
        for input_file in input_files:
            input_path = os.path.join(self.train_dir, input_file)
            try:
                # Read CSV into Polars DataFrame
                df = pl.read_csv(input_path, infer_schema_length=10000)
                
                # Filter for the specific player whose movement we need to predict (case insensitive)
                if "player_to_predict" in df.columns:
                    df = df.filter(
                        pl.col("player_to_predict").cast(pl.Utf8).str.to_lowercase() == "true"
                    )
                
                if df.height > 0:
                    dataframes.append(df)
            except Exception as e:
                print(f"Error loading {input_file}: {e}")

        if not dataframes:
            print("No valid input data found.")
            self.input_sequences = pl.DataFrame()
            return

        # Concatenate all input dataframes
        full_df = pl.concat(dataframes, how="vertical_relaxed")

        # --- Feature Processing (Vectorized Polars Expressions) ---
        
        # Identify ID columns and known non-feature columns to exclude from processing
        id_cols = ["game_id", "play_id", "nfl_id", "frame_id", "player_to_predict", "time", "player_name", "num_frames_output", "ball_land_x", "ball_land_y"]
        feature_cols = [c for c in full_df.columns if c not in id_cols]
        
        expressions = []
        for col in feature_cols:
            # 1. Specific handling for 'player_birth_date': Convert to player age in years
            if col == "player_birth_date":
                # Assuming game date is 2023-09-01 for age calculation
                expr = (pl.lit("2023-09-01").str.to_date() - pl.col(col).str.to_date(format="%Y-%m-%d", strict=False)).dt.total_days() / 365.25
                expressions.append(expr.alias(col))
                continue

            # 2. General handling for string columns (categorical, boolean strings, ranges)
            if full_df[col].dtype == pl.Utf8:
                expr = (
                    # Handle composite values like weight ranges (e.g., "180-200")
                    pl.when(pl.col(col).str.contains(r"^\d+-\d+$")).then(
                        # Simple average of the range
                        (
                            pl.col(col).str.extract(r"(\d+)-(\d+)", 1).cast(pl.Int32) +
                            pl.col(col).str.extract(r"(\d+)-(\d+)", 2).cast(pl.Int32)
                        ) / 2
                    )
                    # Convert common categorical strings to floats (1.0 or 0.0)
                    .when(pl.col(col).str.to_lowercase() == "true").then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "false").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "left").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "right").then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "defense").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "offense").then(1.0)
                    .otherwise(
                        # Fallback: Try casting to float. If conversion fails (is null), use hash.
                        pl.col(col).cast(pl.Float64, strict=False).fill_null(
                            pl.col(col).hash() % 10000 # Deterministic hashing for categorical features
                        )
                    ).cast(pl.Float64).alias(col)
                )
                expressions.append(expr)
            else:
                # 3. Numeric columns: Ensure they are Float64
                expressions.append(pl.col(col).cast(pl.Float64).alias(col))

        # Apply all feature transformations
        full_df = full_df.with_columns(expressions)
        
        # Sort by ID keys and frame_id to ensure correct temporal order within a sequence
        if "frame_id" in full_df.columns:
            full_df = full_df.sort(["game_id", "play_id", "nfl_id", "frame_id"])
        
        # Group features into lists (one list per feature per sequence)
        agg_exprs = [pl.col(c) for c in feature_cols]
        
        # Aggregate the time steps for each feature column into a single list per sequence
        grouped = full_df.group_by(["game_id", "play_id", "nfl_id"], maintain_order=True).agg(agg_exprs)
        
        # The result is a DataFrame where each row is a sequence (a play-player combination)
        # and feature columns contain lists of values over time.
        self.input_sequences = grouped

    def load_output_files(self):
        """
        Loads output CSV files from the training directory using Polars.

        Iterates through files starting with 'output' and ending with '.csv'.
        Extracts the target features ('x' and 'y' coordinates), ensures they are
        of float type, and groups them by (game_id, play_id, nfl_id) to form
        output sequences (lists of targets).
        """
        output_files = sorted([f for f in os.listdir(self.train_dir) if f.startswith('output') and f.endswith('.csv')])
        print(f"Loading {len(output_files)} Output files...")
        
        features_to_keep = ['x', 'y']
        dataframes = []
        
        for output_file in output_files:
            output_path = os.path.join(self.train_dir, output_file)
            try:
                # Select only the required ID columns and target features
                df = pl.read_csv(output_path, columns=['game_id', 'play_id', 'nfl_id'] + features_to_keep, infer_schema_length=10000)
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {output_file}: {e}")

        if not dataframes:
            print("No valid output data found.")
            self.output_sequences = pl.DataFrame()
            return

        full_df = pl.concat(dataframes, how="vertical_relaxed")
        
        # Ensure target features are float type
        full_df = full_df.with_columns([
            pl.col(c).cast(pl.Float64) for c in features_to_keep
        ])
        
        # Sort if frame info is implicit (usually matches input)
        # We don't have frame_id in output usually? Assuming same order.
        
        # Group and aggregate target features into lists
        grouped = full_df.group_by(["game_id", "play_id", "nfl_id"], maintain_order=True).agg([
            pl.col('x'),
            pl.col('y')
        ])
        
        self.output_sequences = grouped

    def get_aligned_data(self):
        """
        Aligns input and output sequences based on common sequence keys.

        Performs an inner join on the grouped input and output data to ensure
        only matching sequences are retained. It then converts the Polars
        DataFrame of feature lists into two NumPy object arrays of sequences
        suitable for Keras training.

        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): Array of input sequences (object array). Each element
                  is a 2D array of shape (seq_len, n_features).
                - y (np.ndarray): Array of output sequences (object array). Each element
                  is a 2D array of shape (seq_len, n_targets).
        """
        self.load_input_files()
        self.load_output_files()

        print("Aligning Input and Output sequences...")
        
        if self.input_sequences is None or self.input_sequences.is_empty():
            print("Input sequences empty.")
            return np.array([]), np.array([])
            
        if self.output_sequences is None or self.output_sequences.is_empty():
            print("Output sequences empty.")
            return np.array([]), np.array([])

        # Inner join on keys to keep only matching sequences
        joined = self.input_sequences.join(
            self.output_sequences, 
            on=["game_id", "play_id", "nfl_id"], 
            how="inner",
            suffix="_out" # Use suffix for output columns in case of collision
        )
        
        print(f"Processing complete.")
        print(f"Total Unique Sequences (Matches): {len(joined)}")

        if len(joined) == 0:
            print("No matching data found.")
            return np.array([]), np.array([])

        # --- Convert from Polars lists to NumPy sequences ---
        
        # Determine the final feature and output columns in the joined DF
        input_cols = [c for c in self.input_sequences.columns if c not in ["game_id", "play_id", "nfl_id"]]
        # Output columns: check for suffix '_out' or use original names (assuming x, y are the target features)
        output_cols = ["x_out", "y_out"] 
        if "x_out" not in joined.columns:
            output_cols = ["x", "y"]
            
        print("Converting to NumPy arrays...")
        
        # Pre-fetch column indices for efficient iteration
        input_col_indices = [joined.columns.index(c) for c in input_cols]
        output_col_indices = [joined.columns.index(c) for c in output_cols]
        
        rows = joined.iter_rows()
        
        X_list = []
        y_list = []
        
        for row in rows:
            # 1. Extract feature sequences for the current row
            # feature_seqs: [ [feat1_t0, feat1_t1...], [feat2_t0, feat2_t1...] ... ]
            feature_seqs = [row[i] for i in input_col_indices]
            
            # 2. Transpose and stack: Polars stores (n_features, n_timesteps), we need (n_timesteps, n_features)
            # zip(*...) transposes the lists for column stacking
            X_seq = list(zip(*feature_seqs))
            X_list.append(X_seq)
            
            # 3. Extract output sequences
            out_seqs = [row[i] for i in output_col_indices]
            y_seq = list(zip(*out_seqs))
            y_list.append(y_seq)
            
        # Create NumPy object arrays (to handle variable-length sequences)
        X = np.array(X_list, dtype=object)
        y = np.array(y_list, dtype=object)
        
        print(f"Initial X shape: {X.shape}")
        print(f"Initial y shape: {y.shape}")
            
        return X, y


class NFLDataSequence(Sequence):
    """Keras Sequence for NFL data with automatic padding of variable-length sequences.

    Inherits from `tensorflow.keras.utils.Sequence` to provide a data generator
    that can be used with Keras models. Handles batching, shuffling, and
    padding of sequences to a uniform length.
    """
    def __init__(self, X, y, batch_size=32, maxlen_x=None, maxlen_y=None, shuffle=True):
        """Initializes the NFLDataSequence.

        Args:
            X (list or np.ndarray): List of input sequences, where each sequence
                is a list of time steps.
            y (list or np.ndarray): List of output sequences, where each sequence
                is a list of time steps.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            maxlen_x (int, optional): Maximum length for input sequences. If None,
                it is calculated from the data. Defaults to None.
            maxlen_y (int, optional): Maximum length for output sequences. If None,
                it is calculated from the data. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the data at the end of
                each epoch. Defaults to True.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        
        # Determine max lengths if not provided
        if maxlen_x is None:
            self.maxlen_x = max(len(seq) for seq in X)
        else:
            self.maxlen_x = maxlen_x
            
        if maxlen_y is None:
            self.maxlen_y = max(len(seq) for seq in y)
        else:
            self.maxlen_y = maxlen_y
        
        print(f"NFLDataSequence initialized: {len(self.X)} samples, batch_size={batch_size}")
        print(f"Max sequence lengths - X: {self.maxlen_x}, y: {self.maxlen_y}")
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Computes the number of batches per epoch.

        Returns:
            int: The number of batches.
        """
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        """Generates one batch of data.

        Args:
            idx (int): The index of the batch.

        Returns:
            tuple: A tuple (X_padded, y_padded) containing the padded input and
                output sequences for the batch.
        """
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get batch data
        batch_X = [self.X[i] for i in batch_indices]
        batch_y = [self.y[i] for i in batch_indices]
        
        # Process X sequences: handle mixed types
        # With Polars preprocessing, data should already be numeric floats
        # But let's ensure it's a list of lists of floats
        
        # batch_X is a list of sequences. Each sequence is a list of frames. Each frame is a list of features.
        # We need to convert this to a 3D numpy array or list of 2D arrays for pad_sequences
        
        # Since we did the conversion in get_aligned_data, batch_X elements should be lists of tuples/lists of floats.
        # We can directly pass this to pad_sequences if they are numeric.
        
        # Use pad_sequences for both X and y
        # pad_sequences expects sequences of shape (n_samples, n_timesteps) for 2D
        # For 3D (n_samples, n_timesteps, n_features), we need to pad manually or use padding='post'
        
        # Method: Pad each sequence to maxlen, filling with zeros
        X_padded = pad_sequences(
            batch_X, 
            maxlen=self.maxlen_x, 
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0
        )
        
        y_padded = pad_sequences(
            batch_y,
            maxlen=self.maxlen_y,
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0
        )
        
        return X_padded, y_padded
    
    def on_epoch_end(self):
        """Updates indexes after each epoch.

        If `self.shuffle` is True, the data indices are shuffled to ensure
        random batch composition in the next epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_tf_datasets(X, y, test_size=0.2, batch_size=32, maxlen_x=None, maxlen_y=None):
    """Splits data into training and validation sets and creates Keras Sequence datasets.

    Uses `train_test_split` to divide the data and then wraps the resulting
    sets in `NFLDataSequence` objects, which handle padding and batching.

    Args:
        X (np.ndarray): Input data (object array of variable-length sequences).
        y (np.ndarray): Output data (object array of variable-length sequences).
        test_size (float, optional): Proportion of the dataset to include in the
            validation split. Defaults to 0.2.
        batch_size (int, optional): Batch size for the datasets. Defaults to 32.
        maxlen_x (int, optional): Maximum length for input sequences. If None,
            the maximum length in the training set will be used. Defaults to None.
        maxlen_y (int, optional): Maximum length for output sequences. If None,
            the maximum length in the training set will be used. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - train_sequence (NFLDataSequence): The training data sequence.
            - val_sequence (NFLDataSequence): The validation data sequence.
            Returns (None, None) if an error occurs.
    """
    print("\n--- Creating Keras Sequence Datasets with Padding ---")
    
    try:
        # Convert object arrays to lists
        X_list = X.tolist()
        y_list = y.tolist()
        
        # Split into train and validation
        print(f"Splitting data (test_size={test_size})...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_list, y_list, 
            test_size=test_size, 
            random_state=42
        )
        
        print(f"Train size: {len(X_train)}")
        print(f"Val size: {len(X_val)}")
        
        # Create Sequence objects
        print("Creating Training Sequence...")
        train_sequence = NFLDataSequence(
            X_train, y_train, 
            batch_size=batch_size,
            maxlen_x=maxlen_x,
            maxlen_y=maxlen_y,
            shuffle=True
        )
        
        print("Creating Validation Sequence...")
        val_sequence = NFLDataSequence(
            X_val, y_val,
            batch_size=batch_size,
            maxlen_x=train_sequence.maxlen_x,  # Use same max lengths as training
            maxlen_y=train_sequence.maxlen_y,
            shuffle=False
        )
        
        print("Sequences created successfully.")
        print(f"Training batches per epoch: {len(train_sequence)}")
        print(f"Validation batches per epoch: {len(val_sequence)}")
        
        return train_sequence, val_sequence

    except Exception as e:
        print(f"Error creating Keras sequences: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    TRAIN_DIR = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    
    loader = NFLDataLoader(TRAIN_DIR)
    X, y = loader.get_aligned_data()

    print("\n--- Final Data Shapes ---")
    print(f"X (Input) Shape: {X.shape}")
    print(f"y (Output) Shape: {y.shape}")

    if len(X) > 0:
        print(f"Sample Input Sequence Length: {len(X[0])}")
        print(f"Sample Output Sequence Length: {len(y[0])}")

    # Create Keras Sequences with padding
    train_seq, val_seq = create_tf_datasets(X, y, batch_size=32)
    
    if train_seq:
        print("\nVerifying Sequence Element:")
        # Get one batch to verify shapes
        x_batch, y_batch = train_seq[0]
        print(f"Batch X shape: {x_batch.shape}")
        print(f"Batch y shape: {y_batch.shape}")
        print(f"Max sequence lengths - X: {train_seq.maxlen_x}, y: {train_seq.maxlen_y}")

    print("\nData loading, alignment, and sequence creation complete.")
