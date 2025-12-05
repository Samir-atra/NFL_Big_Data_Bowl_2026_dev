"""
Module for loading and preprocessing NFL Big Data Bowl data for Unsupervised Learning.

This module defines the UnsupervisedNFLDataLoader, which is designed to maximize
data utilization for pre-training by loading and combining sequence data for ALL
players (regardless of whether they are the 'player_to_predict' in the supervised task).
The data is processed using Polars for efficiency, converted into NumPy object arrays
of variable-length sequences, and can be optionally normalized.

It also provides the UnsupervisedNFLSequence, a Keras Sequence utility for generating
batches for self-supervised tasks like autoencoding or next-step prediction with
on-the-fly padding.
"""
import polars as pl
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence


class UnsupervisedNFLDataLoader:
    """
    Loads and preprocesses all available NFL player tracking data for unsupervised learning.
    
    This loader processes ALL player sequences (player_to_predict=True and False)
    to maximize the amount of training data for representation learning.
    """
    
    def __init__(self):
        """Initializes the loader with an empty sequence container."""
        self.input_sequences = None
        
    def load_files(self, directories, include_labeled=True, include_unlabeled=True, normalize=False):
        """
        Loads and processes input files from specified directories, combining them 
        into sequences of processed features.
        
        Args:
            directories (list): List of directory paths to load input CSV files from.
            include_labeled (bool): If True, sequences where 'player_to_predict' is True are included.
            include_unlabeled (bool): If True, sequences where 'player_to_predict' is False are included.
            normalize (bool): If True, applies Z-score normalization to the features.
        """
        input_dfs = []
        
        print(f"Loading unsupervised data from {len(directories)} directories...")
        print(f"Include labeled: {include_labeled}, Include unlabeled: {include_unlabeled}")
        
        for d in directories:
            if not os.path.exists(d):
                print(f"Warning: Directory not found: {d}")
                continue
                
            input_files = sorted([f for f in os.listdir(d) if f.startswith('input') and f.endswith('.csv')])
            print(f"  Found {len(input_files)} input files in {d}")
            
            for f in input_files:
                try:
                    df = pl.read_csv(os.path.join(d, f), infer_schema_length=10000)
                    
                    initial_rows = len(df)
                    
                    # Filter based on player_to_predict flag
                    if "player_to_predict" in df.columns:
                        if include_labeled and not include_unlabeled:
                            # Only labeled (player_to_predict == True)
                            if df["player_to_predict"].dtype == pl.Boolean:
                                df = df.filter(pl.col("player_to_predict") == True)
                            else:
                                df = df.filter(pl.col("player_to_predict").cast(pl.Utf8).str.to_lowercase() == "true")
                        elif include_unlabeled and not include_labeled:
                            # Only unlabeled (player_to_predict == False)
                            if df["player_to_predict"].dtype == pl.Boolean:
                                df = df.filter(pl.col("player_to_predict") == False)
                            else:
                                df = df.filter(pl.col("player_to_predict").cast(pl.Utf8).str.to_lowercase() == "false")
                        # If both True, include all (no filtering)
                    
                    if len(df) > 0:
                        input_dfs.append(df)
                        print(f"    {f}: {initial_rows} -> {len(df)} rows")
                        
                except Exception as e:
                    print(f"Error loading {f}: {e}")
        
        if not input_dfs:
            print("No data found.")
            self.input_sequences = pl.DataFrame()
            return
        
        # Concatenate all dataframes
        print("Concatenating dataframes...")
        full_input = pl.concat(input_dfs, how="vertical_relaxed")
        
        # Deduplicate - crucial when loading from multiple sources/weeks
        full_input = full_input.unique(subset=["game_id", "play_id", "nfl_id", "frame_id"])
        
        # Process features
        print("Processing features...")
        # Define ID columns to be excluded from feature vector
        id_cols = ["game_id", "play_id", "nfl_id", "frame_id", "player_to_predict", "time"]
        feature_cols = [c for c in full_input.columns if c not in id_cols]
        
        expressions = []
        for col in feature_cols:
            # Feature engineering / conversion logic (must match supervised loader)
            if full_input[col].dtype == pl.Utf8:
                expr = (
                    pl.when(pl.col(col).str.to_lowercase() == "true").then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "false").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "left").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "right").then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "defense").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "offense").then(1.0)
                    .otherwise(
                        # Fallback: cast to float, fill nulls with hash
                        pl.col(col).cast(pl.Float64, strict=False).fill_null(
                            pl.col(col).hash() % 10000
                        )
                    ).cast(pl.Float64).alias(col)
                )
                expressions.append(expr)
            else:
                # Ensure all features are float type
                expressions.append(pl.col(col).cast(pl.Float64).alias(col))
        
        full_input = full_input.with_columns(expressions)
        
        # Sort by frame_id to ensure sequences are temporal
        if "frame_id" in full_input.columns:
            full_input = full_input.sort(["game_id", "play_id", "nfl_id", "frame_id"])
            
        # Normalize features if requested (Z-score)
        if normalize:
            print("Normalizing features (Z-score)...")
            norm_exprs = []
            for col in feature_cols:
                # Check for constant columns to avoid division by zero
                n_unique = full_input.select(pl.col(col).n_unique()).item()
                if n_unique > 1:
                    norm_exprs.append(
                        ((pl.col(col) - pl.col(col).mean()) / (pl.col(col).std() + 1e-8)).alias(col)
                    )
                else:
                    # Keep constant columns as is (mean is the value, std is 0)
                    norm_exprs.append(pl.col(col)) 
            
            if norm_exprs:
                full_input = full_input.with_columns(norm_exprs)
        
        # Group into sequences (one row per unique play-player combination)
        agg_exprs = [pl.col(c) for c in feature_cols]
        self.input_sequences = full_input.group_by(
            ["game_id", "play_id", "nfl_id"], 
            maintain_order=True
        ).agg(agg_exprs)
        
        print(f"Total sequences: {len(self.input_sequences)}")
        
        # Debug sequence lengths
        if len(self.input_sequences) > 0:
            lengths = self.input_sequences.select(pl.col(feature_cols[0]).list.len()).to_series()
            print(f"Sequence length stats: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.2f}")

        
    def get_sequences(self):
        """
        Converts the Polars DataFrame of feature lists into a NumPy object array 
        where each element is a 2D NumPy array (seq_len, n_features).
        
        Returns:
            np.ndarray: Array of input sequences (object array). Returns an empty
                        array if no data was loaded.
        """
        if self.input_sequences is None or self.input_sequences.is_empty():
            return np.array([])
        
        print("Converting to NumPy arrays...")
        
        # Get feature columns (exclude keys)
        input_cols = [c for c in self.input_sequences.columns 
                     if c not in ["game_id", "play_id", "nfl_id"]]
        
        # Setup for row iteration
        input_col_indices = [self.input_sequences.columns.index(c) for c in input_cols]
        rows = self.input_sequences.iter_rows()
        
        X_list = []
        for row in rows:
            feature_seqs = [row[i] for i in input_col_indices]
            # Transpose the feature lists and stack them into a (T, F) array
            X_seq = np.array(list(zip(*feature_seqs)), dtype=np.float32)
            if len(X_seq) > 0:
                X_list.append(X_seq)
        
        X = np.array(X_list, dtype=object)
        print(f"Loaded {len(X)} sequences")
        
        return X


class UnsupervisedNFLSequence(Sequence):
    """
    Keras Sequence for unsupervised learning on NFL data.
    
    Generates batches for self-supervised tasks with on-the-fly padding:
    - 'autoencoder': Input and output are the same padded sequence (reconstruction).
    - 'next_step': Input is sequence[:-n], output is sequence[-n:] (predicting n steps ahead).
    """
    
    def __init__(self, X, batch_size=32, maxlen=None, shuffle=True, 
                 task='autoencoder', prediction_steps=1):
        """
        Initializes the sequence generator.
        
        Args:
            X (np.ndarray): Input sequences (NumPy object array of 2D arrays).
            batch_size (int): Number of sequences per batch.
            maxlen (int, optional): Maximum sequence length to pad to. If None,
                                    it is determined by the longest sequence in X.
            shuffle (bool): If True, shuffles data indices after each epoch.
            task (str): The self-supervised task ('autoencoder' or 'next_step').
            prediction_steps (int): For 'next_step' task, the number of steps to predict.
                                    The input length will be maxlen - prediction_steps.
        
        Raises:
            ValueError: If maxlen is not greater than prediction_steps for 'next_step' task.
        """
        self.X = X
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.task = task
        self.prediction_steps = prediction_steps
        self.indices = np.arange(len(self.X))
        
        # Determine max length
        if maxlen is None:
            self.maxlen = max(len(seq) for seq in X)
        else:
            self.maxlen = maxlen
        
        print(f"UnsupervisedNFLSequence initialized:")
        print(f"  Samples: {len(self.X)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max length: {self.maxlen}")
        print(f"  Task: {task}")
        
        if self.task == 'next_step':
            if self.maxlen <= self.prediction_steps:
                raise ValueError(
                    f"Invalid configuration: maxlen ({self.maxlen}) must be greater than "
                    f"prediction_steps ({self.prediction_steps}) for 'next_step' task.\n"
                    f"Please increase maxlen or decrease prediction_steps."
                )
            
            # Ensure we don't produce 0-length sequences even if maxlen is technically larger but close
            effective_len = self.maxlen - self.prediction_steps
            if effective_len < 1:
                 raise ValueError(f"Effective sequence length (maxlen - prediction_steps) is {effective_len}, must be >= 1.")

        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Returns the number of batches per epoch (required by Keras Sequence)."""
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        """Generates one batch of data (required by Keras Sequence)."""
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Filter out empty sequences to prevent pad_sequences errors
        batch_X = [self.X[i] for i in batch_indices if len(self.X[i]) > 0]
        
        if not batch_X:
            # Return dummy batch with correct shape to avoid crashing the training loop
            if len(self.X) > 0 and len(self.X[0]) > 0:
                 feat_dim = self.X[0].shape[1]
                 # The output shape depends on the task, but for safety, return shape matching input/output
                 return np.zeros((0, self.maxlen, feat_dim)), np.zeros((0, self.maxlen, feat_dim))
            return np.array([]), np.array([])
        
        if self.task == 'autoencoder':
            # Autoencoder: Input X is equal to Output Y (reconstruction target)
            X_padded = pad_sequences(
                batch_X,
                maxlen=self.maxlen,
                dtype='float32',
                padding='post',
                truncating='post',
                value=0.0
            )
            return X_padded, X_padded
            
        elif self.task == 'next_step':
            # Next-step prediction: Input X is observation, Output Y is prediction target
            batch_X_input = []
            batch_y_output = []
            
            for seq in batch_X:
                if len(seq) > self.prediction_steps:
                    # X is the history, Y is the future steps
                    batch_X_input.append(seq[:-self.prediction_steps])
                    batch_y_output.append(seq[-self.prediction_steps:])
                else:
                    # If sequence too short, this scenario should be rare/filtered, but handle defensively
                    # Use full sequence for both, knowing the model will likely learn nothing useful
                    batch_X_input.append(seq)
                    batch_y_output.append(seq)
            
            # Pad the input sequences to maxlen - prediction_steps
            X_padded = pad_sequences(
                batch_X_input,
                maxlen=self.maxlen - self.prediction_steps,
                dtype='float32',
                padding='post',
                truncating='post',
                value=0.0
            )
            
            # Pad the output sequences to prediction_steps
            y_padded = pad_sequences(
                batch_y_output,
                maxlen=self.prediction_steps,
                dtype='float32',
                padding='post',
                truncating='post',
                value=0.0
            )
            
            return X_padded, y_padded
    
    def on_epoch_end(self):
        """Shuffles the indices after an epoch if shuffle=True (required by Keras Sequence)."""
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == "__main__":
    # Test the loader
    PREDICTION_TRAIN_DIR = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    
    print("=== Testing Unsupervised Data Loader ===\n")
    
    # Test 1: Load only unlabeled data
    print("Test 1: Loading UNLABELED data only")
    loader = UnsupervisedNFLDataLoader()
    loader.load_files([PREDICTION_TRAIN_DIR], include_labeled=False, include_unlabeled=True)
    X_unlabeled = loader.get_sequences()
    print(f"Unlabeled sequences: {len(X_unlabeled)}\n")
    
    # Test 2: Load ALL data
    print("Test 2: Loading ALL data (labeled + unlabeled)")
    loader_all = UnsupervisedNFLDataLoader()
    loader_all.load_files([PREDICTION_TRAIN_DIR], include_labeled=True, include_unlabeled=True)
    X_all = loader_all.get_sequences()
    print(f"Total sequences: {len(X_all)}\n")
    
    if len(X_all) > 0:
        print(f"Sample sequence length: {len(X_all[0])}")
        print(f"Sample features per timestep: {len(X_all[0][0])}")
        
        # Test sequence generators
        print("\n=== Testing Sequence Generators ===")
        
        print("\nAutoencoder sequence:")
        ae_seq = UnsupervisedNFLSequence(X_all[:1000], batch_size=32, task='autoencoder')
        x_batch, y_batch = ae_seq[0]
        print(f"Input shape: {x_batch.shape}")
        print(f"Output shape: {y_batch.shape}")
        print(f"Are input and output same? {np.array_equal(x_batch, y_batch)}")
        
        print("\nNext-step prediction sequence:")
        ns_seq = UnsupervisedNFLSequence(X_all[:1000], batch_size=32, task='next_step', prediction_steps=5)
        x_batch, y_batch = ns_seq[0]
        print(f"Input shape: {x_batch.shape}")
        print(f"Output shape: {y_batch.shape}")

    # Test 3: Load with normalization
    print("\nTest 3: Loading with NORMALIZATION")
    loader_norm = UnsupervisedNFLDataLoader()
    loader_norm.load_files([PREDICTION_TRAIN_DIR], include_labeled=True, include_unlabeled=True, normalize=True)
    X_norm = loader_norm.get_sequences()
    
    if len(X_norm) > 0:
        print(f"Sample normalized sequence (first feature of first step): {X_norm[0][0][0]}")
