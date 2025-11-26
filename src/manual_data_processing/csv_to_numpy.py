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
        input_sequences (pl.DataFrame): DataFrame containing input sequences.
        output_sequences (pl.DataFrame): DataFrame containing output sequences.
    """
    def __init__(self, train_dir):
        self.train_dir = train_dir
        self.input_sequences = None
        self.output_sequences = None

    def load_input_files(self):
        """Loads and filters input CSV files from the training directory using Polars.

        Iterates through files starting with 'input' and ending with '.csv'.
        Filters rows where 'player_to_predict' is True and groups them by
        (game_id, play_id, nfl_id) to form sequences.
        """
        input_files = sorted([f for f in os.listdir(self.train_dir) if f.startswith('input') and f.endswith('.csv')])
        print(f"Loading and filtering {len(input_files)} Input files...")
        
        dataframes = []
        for input_file in input_files:
            input_path = os.path.join(self.train_dir, input_file)
            try:
                # Lazy load for efficiency, though read_csv is fine for smaller files
                # Using read_csv to ensure we catch errors immediately
                df = pl.read_csv(input_path, infer_schema_length=10000)
                
                # Filter for player_to_predict == True (case insensitive)
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

        # Process columns (Vectorized)
        # Handle Booleans, Directions, Sides, etc.
        
        # Helper expression for boolean strings
        def to_bool_float(col_name):
            return (
                pl.when(pl.col(col_name).cast(pl.Utf8).str.to_lowercase() == "true").then(1.0)
                .when(pl.col(col_name).cast(pl.Utf8).str.to_lowercase() == "false").then(0.0)
                .otherwise(0.0) # Default or handle errors
            )

        # Helper for direction
        def to_dir_float(col_name):
            return (
                pl.when(pl.col(col_name).cast(pl.Utf8).str.to_lowercase() == "left").then(0.0)
                .when(pl.col(col_name).cast(pl.Utf8).str.to_lowercase() == "right").then(1.0)
                .otherwise(0.0)
            )

        # Helper for side
        def to_side_float(col_name):
            return (
                pl.when(pl.col(col_name).cast(pl.Utf8).str.to_lowercase() == "defense").then(0.0)
                .when(pl.col(col_name).cast(pl.Utf8).str.to_lowercase() == "offense").then(1.0)
                .otherwise(0.0)
            )
            
        # Apply transformations
        # We need to identify columns to transform. Based on previous code:
        # Booleans: player_to_predict (already filtered, but maybe others?)
        # Direction: play_direction? (Not explicitly named in previous code but handled in generic process_value)
        # Side: player_side?
        
        # For generic handling, we can inspect types, but for performance, explicit is better.
        # Let's assume standard columns or iterate if needed.
        # The previous code iterated every cell. Here we want vectorization.
        # We will cast all remaining columns to float, hashing strings if needed.
        
        # Identify ID columns to exclude from feature processing
        id_cols = ["game_id", "play_id", "nfl_id", "frame_id", "player_to_predict", "time"]
        feature_cols = [c for c in full_df.columns if c not in id_cols]
        
        expressions = []
        for col in feature_cols:
            # Check if column is string type
            if full_df[col].dtype == pl.Utf8:
                # Try specific conversions first
                # We can't easily check content of every row efficiently without scanning
                # So we apply a complex expression:
                # If 'true'/'false' -> 1/0
                # If 'left'/'right' -> 0/1
                # If 'defense'/'offense' -> 0/1
                # Else try cast float
                # Else hash
                
                expr = (
                    pl.when(pl.col(col).str.to_lowercase() == "true").then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "false").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "left").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "right").then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "defense").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "offense").then(1.0)
                    .otherwise(
                        # Try cast to float, if null (failed), then hash
                        pl.col(col).cast(pl.Float64, strict=False).fill_null(
                            pl.col(col).hash() % 10000
                        )
                    ).cast(pl.Float64).alias(col)
                )
                expressions.append(expr)
            else:
                # Already numeric (int or float), cast to float
                expressions.append(pl.col(col).cast(pl.Float64).alias(col))

        # Select IDs and processed features
        full_df = full_df.with_columns(expressions)
        
        # Group by keys and aggregate into lists
        # We assume the order is defined by frame_id or file order. 
        # If frame_id exists, sort by it.
        if "frame_id" in full_df.columns:
            full_df = full_df.sort(["game_id", "play_id", "nfl_id", "frame_id"])
        
        # Group and aggregate features into lists
        # We want a list of lists (sequence of steps, where each step is a list of features)
        # Polars agg_list creates a list of values for a column.
        # We need to combine these columns into a single "features" column which is a list of lists?
        # Or just keep them as separate columns of lists.
        # The previous code produced: [[f1, f2, ...], [f1, f2, ...], ...] for each sequence.
        
        # Let's aggregate each feature column into a list
        agg_exprs = [pl.col(c) for c in feature_cols]
        
        grouped = full_df.group_by(["game_id", "play_id", "nfl_id"], maintain_order=True).agg(agg_exprs)
        
        # Now we have:
        # game_id, play_id, nfl_id, col1_list, col2_list, ...
        # We need to transpose this to:
        # game_id, play_id, nfl_id, [[col1_t0, col2_t0, ...], [col1_t1, col2_t1, ...]]
        # This is hard in Polars directly.
        # Easier: Convert to numpy/pandas later or iterate.
        
        # Actually, for Keras, we usually want (samples, timesteps, features).
        # If we have separate columns of lists:
        # col1: [t0, t1, t2]
        # col2: [t0, t1, t2]
        # We can stack them.
        
        self.input_sequences = grouped

    def load_output_files(self):
        """Loads output CSV files from the training directory using Polars.

        Iterates through files starting with 'output' and ending with '.csv'.
        Extracts 'x' and 'y' features, grouping them by (game_id, play_id, nfl_id)
        to form sequences.
        """
        output_files = sorted([f for f in os.listdir(self.train_dir) if f.startswith('output') and f.endswith('.csv')])
        print(f"Loading {len(output_files)} Output files...")
        
        features_to_keep = ['x', 'y']
        dataframes = []
        
        for output_file in output_files:
            output_path = os.path.join(self.train_dir, output_file)
            try:
                df = pl.read_csv(output_path, columns=['game_id', 'play_id', 'nfl_id'] + features_to_keep, infer_schema_length=10000)
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {output_file}: {e}")

        if not dataframes:
            print("No valid output data found.")
            self.output_sequences = pl.DataFrame()
            return

        full_df = pl.concat(dataframes, how="vertical_relaxed")
        
        # Ensure float type
        full_df = full_df.with_columns([
            pl.col(c).cast(pl.Float64) for c in features_to_keep
        ])
        
        # Sort if frame info is implicit (usually matches input)
        # We don't have frame_id in output usually? Assuming same order.
        # Ideally we should sort by something, but without frame_id we rely on file order.
        
        grouped = full_df.group_by(["game_id", "play_id", "nfl_id"], maintain_order=True).agg([
            pl.col('x'),
            pl.col('y')
        ])
        
        self.output_sequences = grouped

    def get_aligned_data(self):
        """Aligns input and output sequences based on common keys.

        Loads both input and output files, finds the intersection of keys,
        and creates aligned lists of sequences.

        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): Array of input sequences (object array).
                - y (np.ndarray): Array of output sequences (object array).
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

        # Join on keys
        # Inner join to keep only matching sequences
        joined = self.input_sequences.join(
            self.output_sequences, 
            on=["game_id", "play_id", "nfl_id"], 
            how="inner",
            suffix="_out"
        )
        
        print(f"Processing complete.")
        print(f"Total Unique Sequences (Matches): {len(joined)}")

        if len(joined) == 0:
            print("No matching data found.")
            return np.array([]), np.array([])

        # Convert to the format expected by NFLDataSequence
        # X: list of [ [f1, f2, ...], [f1, f2, ...] ]
        # y: list of [ [x, y], [x, y] ... ]
        
        # The joined dataframe has columns:
        # game_id, play_id, nfl_id, feat1_list, feat2_list, ..., x_list, y_list
        
        # We need to identify feature columns vs output columns
        # Output columns are 'x' and 'y' (from output_sequences, might be renamed if collision)
        # Actually, input also has 'x' and 'y' usually.
        # In load_output_files, we aggregated 'x' and 'y'.
        # In load_input_files, we aggregated all features.
        # If input has 'x', 'y', they will collide.
        # The join suffix="_out" handles this. Output cols will be 'x_out', 'y_out'.
        
        # Input feature columns: all columns from input_sequences except keys
        input_cols = [c for c in self.input_sequences.columns if c not in ["game_id", "play_id", "nfl_id"]]
        output_cols = ["x_out" if "x" in input_cols else "x", "y_out" if "y" in input_cols else "y"]
        
        # Check if output cols exist
        if output_cols[0] not in joined.columns:
            # Maybe input didn't have x/y, so no suffix
            output_cols = ["x", "y"]
            
        # Convert to numpy
        # This is the heavy part.
        # We can iterate rows or use map_elements?
        # Ideally we want to stack the feature lists.
        
        # Let's extract input features as a list of arrays
        # Each row i has [feat1_seq, feat2_seq, ...]
        # We want [[feat1_t0, feat2_t0], [feat1_t1, feat2_t1], ...]
        
        # Efficient way:
        # 1. Convert relevant columns to a dict of lists or similar
        # 2. Iterate and stack
        
        print("Converting to NumPy arrays...")
        
        # Extract input data
        # shape: (n_samples, n_features, n_timesteps) roughly, but variable timesteps
        # We want (n_samples, n_timesteps, n_features)
        
        # Get all input feature lists as a list of lists of lists?
        # joined.select(input_cols).to_dict(as_series=False) gives {col: [seq1, seq2...]}
        
        # This might be memory intensive.
        # Let's try row iteration with a generator or list comp
        
        # Pre-fetch column indices for speed
        input_col_indices = [joined.columns.index(c) for c in input_cols]
        output_col_indices = [joined.columns.index(c) for c in output_cols]
        
        rows = joined.iter_rows()
        
        X_list = []
        y_list = []
        
        for row in rows:
            # Input
            # row[i] is a list of values for feature i for this sequence
            # We want to stack them: [[val_0_0, val_1_0...], [val_0_1, val_1_1...]]
            # Zip is useful here
            
            # Get all feature sequences for this row
            feature_seqs = [row[i] for i in input_col_indices]
            # feature_seqs is [ [t0, t1...], [t0, t1...] ... ] (n_features, n_timesteps)
            # We want (n_timesteps, n_features)
            # zip(*feature_seqs) does exactly this transpose
            
            # Note: Polars lists might be None if empty? Assuming data is clean.
            # Also assuming all feature lists have same length (they should if from same rows)
            
            X_seq = list(zip(*feature_seqs))
            X_list.append(X_seq)
            
            # Output
            out_seqs = [row[i] for i in output_col_indices]
            y_seq = list(zip(*out_seqs))
            y_list.append(y_seq)
            
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


def create_tf_datasets(X, y, test_size=0.2, batch_size=32, maxlen_x=10, maxlen_y=10):
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
            auto-detects from the training set. Defaults to 10.
        maxlen_y (int, optional): Maximum length for output sequences. If None,
            auto-detects from the training set. Defaults to 10.

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
