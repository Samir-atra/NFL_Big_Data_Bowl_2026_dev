"""
Module for end-to-end data loading, preprocessing, and batching for the
NFL Big Data Bowl 2026 sequence prediction task.

It uses the Polars library for efficient data loading and manipulation,
TensorFlow's Keras Sequence utility for memory-efficient training with
variable-length sequences, and implements feature engineering,
normalization, and sequence padding.
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
    """
    Loads and processes NFL Big Data Bowl 2026 data from CSV files using Polars.
    Filters input data for 'player_to_predict' == True and aligns with output data.
    
    Selected Input Features: All available features after processing.
    Selected Output Targets: ['x', 'y']
    """
    def __init__(self, train_dir):
        """
        Initializes the DataLoader with the training data directory.

        Args:
            train_dir (str): Path to the directory containing input and output CSV files.
        """
        self.train_dir = train_dir
        self.input_df = None
        self.output_df = None
        self.stats = {}

    def process_features(self, df):
        """
        Processes features using Polars expressions to convert categorical/string
        data (like boolean flags or 'offense'/'defense') into numeric float types.
        It uses deterministic hashing as a fallback for unhandled string values.

        Args:
            df (pl.DataFrame): The Polars DataFrame containing raw data.

        Returns:
            list: A list of Polars expressions for feature selection and transformation.
        """
        # Define ID columns to exclude from feature processing
        id_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'player_to_predict', 'time']
        
        # Identify feature columns
        feature_cols = [col for col in df.columns if col not in id_cols]
        
        expressions = []
        for col in feature_cols:
            # Check if column is string type
            if df[col].dtype == pl.Utf8:
                expr = (
                    pl.when(pl.col(col).str.to_lowercase() == "true").then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "false").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "left").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "right").then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "defense").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "offense").then(1.0)
                    .otherwise(
                        # Try to cast to float, if fails (null), use hash
                        # Use seed=42 for deterministic hashing
                        pl.col(col).cast(pl.Float64, strict=False).fill_null(
                            pl.col(col).hash(seed=42).mod(10000).cast(pl.Float64)
                        )
                    ).cast(pl.Float64).alias(col)
                )
                expressions.append(expr)
            else:
                # Cast numeric columns to Float64
                expressions.append(pl.col(col).cast(pl.Float64).alias(col))
                
        return expressions

    def load_input_files(self):
        """
        Loads and filters all input CSV files using Polars.
        It filters for the rows corresponding to the 'player_to_predict' == True
        and then processes the features.
        """
        input_files = sorted([f for f in os.listdir(self.train_dir) if f.startswith('input') and f.endswith('.csv')])
        print(f"Loading and filtering {len(input_files)} Input files...")
        
        dfs = []
        for f in input_files:
            try:
                path = os.path.join(self.train_dir, f)
                # Read CSV
                df = pl.read_csv(path, infer_schema_length=10000)
                
                # Filter for player_to_predict == True
                if "player_to_predict" in df.columns:
                    # Handle both boolean and string representations of 'True'
                    if df["player_to_predict"].dtype == pl.Boolean:
                        df = df.filter(pl.col("player_to_predict") == True)
                    else:
                        df = df.filter(pl.col("player_to_predict").cast(pl.Utf8).str.to_lowercase() == "true")
                
                if len(df) > 0:
                    dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        if not dfs:
            print("No input data found.")
            self.input_df = pl.DataFrame()
            return

        # Concatenate all input dataframes
        self.input_df = pl.concat(dfs, how="vertical_relaxed")
        
        # Process features
        print("Processing input features...")
        feature_exprs = self.process_features(self.input_df)
        
        # Keep ID columns and processed features
        id_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id']
        # Ensure we only select columns that exist
        existing_id_cols = [c for c in id_cols if c in self.input_df.columns]
        
        self.input_df = self.input_df.select(
            [pl.col(c) for c in existing_id_cols] + feature_exprs
        )

    def load_output_files(self):
        """
        Loads all output CSV files using Polars and selects the target features,
        specifically ['x', 'y'] for prediction.
        """
        output_files = sorted([f for f in os.listdir(self.train_dir) if f.startswith('output') and f.endswith('.csv')])
        print(f"Loading {len(output_files)} Output files...")
        
        dfs = []
        features_to_keep = ['x', 'y']
        
        for f in output_files:
            try:
                path = os.path.join(self.train_dir, f)
                df = pl.read_csv(path, infer_schema_length=10000)
                
                # Select IDs and target features
                cols_to_select = ['game_id', 'play_id', 'nfl_id', 'frame_id'] + features_to_keep
                # Ensure all columns exist
                existing_cols = [c for c in cols_to_select if c in df.columns]
                
                if len(existing_cols) == len(cols_to_select):
                    dfs.append(df.select(existing_cols))
                else:
                    print(f"Missing columns in {f}. Found: {existing_cols}")
                    
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if not dfs:
            print("No output data found.")
            self.output_df = pl.DataFrame()
            return

        self.output_df = pl.concat(dfs, how="vertical_relaxed")
        
        # Cast targets to float
        self.output_df = self.output_df.with_columns([
            pl.col(c).cast(pl.Float64) for c in features_to_keep
        ])

    def normalize_data(self, df, feature_cols):
        """
        Normalizes the specified features in the DataFrame using Z-score (standard
        normalization: (value - mean) / std).

        Args:
            df (pl.DataFrame): The DataFrame to normalize.
            feature_cols (list): List of column names to normalize.

        Returns:
            tuple: A tuple containing (normalized_df, statistics_dict).
        """
        print("Normalizing features...")
        stats = {}
        exprs = []
        
        for col in feature_cols:
            mean = df.select(pl.col(col).mean()).item()
            std = df.select(pl.col(col).std()).item()
            
            # Avoid division by zero
            if std == 0 or std is None:
                std = 1.0
            if mean is None:
                mean = 0.0
                
            stats[col] = {'mean': mean, 'std': std}
            exprs.append(((pl.col(col) - mean) / std).alias(col))
            
        return df.with_columns(exprs), stats

    def get_aligned_data(self, normalize=False):
        """
        Loads input and output data, aligns them by common sequence identifiers,
        performs optional normalization, and groups the frames into sequences.

        Args:
            normalize (bool, optional): If True, normalizes the input features (X)
                                        using Z-score. Defaults to False.

        Returns:
            tuple: (X, y) where X and y are NumPy object arrays of sequences.
                   X shape: (n_sequences,) containing arrays of shape (seq_len, n_features)
                   y shape: (n_sequences,) containing arrays of shape (seq_len, n_targets)
        """
        self.load_input_files()
        self.load_output_files()
        
        if self.input_df is None or self.input_df.is_empty() or \
           self.output_df is None or self.output_df.is_empty():
            return np.array([]), np.array([])

        print("Aligning Input and Output sequences...")
        
        # Rename output columns to avoid collision with input columns
        output_features = ['x', 'y']
        rename_map = {c: f"target_{c}" for c in output_features}
        self.output_df = self.output_df.rename(rename_map)
        
        # Join input and output on IDs
        joined_df = self.input_df.join(
            self.output_df,
            on=['game_id', 'play_id', 'nfl_id', 'frame_id'],
            how='inner'
        )
        
        # Sort by frame_id to ensure temporal order
        joined_df = joined_df.sort(['game_id', 'play_id', 'nfl_id', 'frame_id'])
        
        # Identify feature columns (from input) and target columns (from output)
        id_cols = {'game_id', 'play_id', 'nfl_id', 'frame_id'}
        input_cols = [c for c in self.input_df.columns if c not in id_cols]
        target_cols = [f"target_{c}" for c in output_features]
        
        # Normalize if requested
        if normalize:
            # Normalize input features
            normalized_df, self.stats = self.normalize_data(joined_df, input_cols)
            joined_df = normalized_df
            
        # Group by sequence
        print("Grouping sequences...")
        
        # Aggregation expressions
        input_agg = [pl.col(c) for c in input_cols]
        target_agg = [pl.col(c) for c in target_cols]
        
        sequences_df = joined_df.group_by(['game_id', 'play_id', 'nfl_id'], maintain_order=True).agg(
            input_agg + target_agg
        )
        
        print(f"Total Unique Sequences: {len(sequences_df)}")
        
        # Convert to NumPy object arrays
        X_list = []
        y_list = []
        
        # Iterate rows and stack features
        rows = sequences_df.iter_rows(named=True)
        for row in rows:
            # Check sequence length
            seq_len = len(row[input_cols[0]])
            if seq_len == 0:
                continue
                
            # Stack features: (seq_len, n_features)
            input_seq = np.column_stack([row[c] for c in input_cols])
            target_seq = np.column_stack([row[c] for c in target_cols])
            
            X_list.append(input_seq)
            y_list.append(target_seq)
            
        X = np.array(X_list, dtype=object)
        y = np.array(y_list, dtype=object)
        
        print(f"Initial X shape: {X.shape}")
        print(f"Initial y shape: {y.shape}")
        
        return X, y


class NFLDataSequence(Sequence):
    """
    Keras Sequence for NFL data with automatic padding of variable-length sequences.
    
    This class is essential for training sequence models (like RNNs/LSTMs) when
    using variable-length sequences, as it handles batch generation and padding
    on the fly, minimizing memory usage.
    """
    def __init__(self, X, y, batch_size=64, maxlen_x=10, maxlen_y=10, shuffle=False):
        """
        Initializes the sequence generator.

        Args:
            X (list/array): Input sequences (list of 2D arrays).
            y (list/array): Output sequences (list of 2D arrays).
            batch_size (int): The number of sequences per batch.
            maxlen_x (int, optional): Maximum length for input sequences. If None,
                                      it is determined by the longest sequence in X.
            maxlen_y (int, optional): Maximum length for output sequences. If None,
                                      it is determined by the longest sequence in y.
            shuffle (bool): Whether to shuffle data at the end of each epoch.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        
        # Determine max lengths if not provided
        if maxlen_x is None:
            # Calculate max length across all sequences in X
            self.maxlen_x = max(len(seq) for seq in X)
        else:
            self.maxlen_x = maxlen_x
            
        if maxlen_y is None:
            # Calculate max length across all sequences in y
            self.maxlen_y = max(len(seq) for seq in y)
        else:
            self.maxlen_y = maxlen_y
        
        print(f"NFLDataSequence initialized: {len(self.X)} samples, batch_size={batch_size}")
        print(f"Max sequence lengths - X: {self.maxlen_x}, y: {self.maxlen_y}")
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch (required by Keras Sequence protocol)."""
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Generates one batch of padded data (required by Keras Sequence protocol).

        Args:
            idx (int): The index of the batch.

        Returns:
            tuple: (X_padded, y_padded) - a batch of input and target arrays.
        """
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get batch data (sequences of different lengths)
        batch_X = [self.X[i] for i in batch_indices]
        batch_y = [self.y[i] for i in batch_indices]
        
        # Pad sequences to a uniform length (maxlen_x and maxlen_y)
        X_padded = pad_sequences(
            batch_X, 
            maxlen=self.maxlen_x, 
            dtype='float32',
            padding='post', # Padding zeros after the sequence
            truncating='post', # Truncate sequences longer than maxlen from the end
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
        """
        Callback used by Keras to shuffle indices after each epoch when shuffle=True.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_tf_datasets(X, y, test_size=0.2, batch_size=64, maxlen_x=10, maxlen_y=10):
    """
    Splits the NumPy sequence arrays into training and validation sets, and then
    wraps them using the NFLDataSequence (a Keras Sequence) for efficient,
    on-the-fly batching and padding during model training.

    Args:
        X (np.ndarray): Input sequences from NFLDataLoader.
        y (np.ndarray): Output sequences from NFLDataLoader.
        test_size (float, optional): Proportion of data to use for validation. Defaults to 0.2.
        batch_size (int, optional): The batch size for the Keras Sequence. Defaults to 64.
        maxlen_x (int, optional): Maximum length for input sequences. Defaults to 10.
        maxlen_y (int, optional): Maximum length for output sequences. Defaults to 10.

    Returns:
        tuple: (train_sequence, val_sequence) or (None, None) on error.
    """
    print("\n--- Creating Keras Sequence Datasets with Padding ---")
    
    try:
        # Convert object arrays to lists for splitting
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
            shuffle=False
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
    
    # Check if directory exists
    if not os.path.exists(TRAIN_DIR):
        print(f"Warning: Directory {TRAIN_DIR} does not exist. Using current directory for testing.")
        TRAIN_DIR = "."

    loader = NFLDataLoader(TRAIN_DIR)
    # Enable normalization
    X, y = loader.get_aligned_data(normalize=True)

    print("\n--- Final Data Shapes ---")
    print(f"X (Input) Shape: {X.shape}")
    print(f"y (Output) Shape: {y.shape}")

    if len(X) > 0:
        print(f"Sample Input Sequence Length: {len(X[0])}")
        print(f"Sample Output Sequence Length: {len(y[0])}")
        print(f"Input Features: {X[0].shape[1]}")
        print(f"Output Features: {y[0].shape[1]}")

    # Create Keras Sequences with padding
    if len(X) > 0:
        train_seq, val_seq = create_tf_datasets(X, y, batch_size=32)
        
        if train_seq:
            print("\nVerifying Sequence Element:")
            # Get one batch to verify shapes
            x_batch, y_batch = train_seq[0]
            print(f"Batch X shape: {x_batch.shape}")
            print(f"Batch y shape: {y_batch.shape}")
            print(f"Max sequence lengths - X: {train_seq.maxlen_x}, y: {train_seq.maxlen_y}")

    print("\nData loading, alignment, and sequence creation complete.")
