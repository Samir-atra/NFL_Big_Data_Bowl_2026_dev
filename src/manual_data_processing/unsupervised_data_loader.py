import polars as pl
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence


class UnsupervisedNFLDataLoader:
    """Loads NFL data for unsupervised learning (no trajectory labels needed).
    
    This loader processes ALL player sequences (player_to_predict=True and False)
    to maximize the amount of training data for representation learning.
    """
    
    def __init__(self):
        self.input_sequences = None
        
    def load_files(self, directories, include_labeled=True, include_unlabeled=True, normalize=False):
        """Load input files from specified directories.
        
        Args:
            directories (list): List of directory paths to load from
            include_labeled (bool): Include player_to_predict=True sequences
            include_unlabeled (bool): Include player_to_predict=False sequences
            normalize (bool): Whether to normalize features (Z-score)
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
                            # Only labeled
                            if df["player_to_predict"].dtype == pl.Boolean:
                                df = df.filter(pl.col("player_to_predict") == True)
                            else:
                                df = df.filter(pl.col("player_to_predict").cast(pl.Utf8).str.to_lowercase() == "true")
                        elif include_unlabeled and not include_labeled:
                            # Only unlabeled
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
        
        # Deduplicate
        full_input = full_input.unique(subset=["game_id", "play_id", "nfl_id", "frame_id"])
        
        # Process features
        print("Processing features...")
        id_cols = ["game_id", "play_id", "nfl_id", "frame_id", "player_to_predict", "time"]
        feature_cols = [c for c in full_input.columns if c not in id_cols]
        
        expressions = []
        for col in feature_cols:
            if full_input[col].dtype == pl.Utf8:
                expr = (
                    pl.when(pl.col(col).str.to_lowercase() == "true").then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "false").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "left").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "right").then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "defense").then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "offense").then(1.0)
                    .otherwise(
                        pl.col(col).cast(pl.Float64, strict=False).fill_null(
                            pl.col(col).hash() % 10000
                        )
                    ).cast(pl.Float64).alias(col)
                )
                expressions.append(expr)
            else:
                expressions.append(pl.col(col).cast(pl.Float64).alias(col))
        
        full_input = full_input.with_columns(expressions)
        
        # Sort by frame_id
        if "frame_id" in full_input.columns:
            full_input = full_input.sort(["game_id", "play_id", "nfl_id", "frame_id"])
            
        # Normalize features if requested
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
            
            if norm_exprs:
                full_input = full_input.with_columns(norm_exprs)
        
        # Group into sequences
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
        """Convert sequences to numpy arrays.
        
        Returns:
            np.ndarray: Array of input sequences (object array)
        """
        if self.input_sequences is None or self.input_sequences.is_empty():
            return np.array([])
        
        print("Converting to NumPy arrays...")
        
        # Get feature columns (exclude keys)
        input_cols = [c for c in self.input_sequences.columns 
                     if c not in ["game_id", "play_id", "nfl_id"]]
        
        # Convert to sequences
        input_col_indices = [self.input_sequences.columns.index(c) for c in input_cols]
        rows = self.input_sequences.iter_rows()
        
        X_list = []
        for row in rows:
            feature_seqs = [row[i] for i in input_col_indices]
            # Convert to numpy array to ensure consistent shape (T, F)
            X_seq = np.array(list(zip(*feature_seqs)), dtype=np.float32)
            if len(X_seq) > 0:
                X_list.append(X_seq)
        
        X = np.array(X_list, dtype=object)
        print(f"Loaded {len(X)} sequences")
        
        return X


class UnsupervisedNFLSequence(Sequence):
    """Keras Sequence for unsupervised learning on NFL data.
    
    For autoencoder: input and output are the same (reconstruction)
    For next-step prediction: input is sequence[:-n], output is sequence[n:]
    """
    
    def __init__(self, X, batch_size=32, maxlen=None, shuffle=True, 
                 task='autoencoder', prediction_steps=1):
        """Initialize the sequence.
        
        Args:
            X: Input sequences
            batch_size: Batch size
            maxlen: Maximum sequence length (auto-detect if None)
            shuffle: Whether to shuffle
            task: 'autoencoder' or 'next_step'
            prediction_steps: For next_step, how many steps ahead to predict
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
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Filter out empty sequences to prevent pad_sequences errors
        batch_X = [self.X[i] for i in batch_indices if len(self.X[i]) > 0]
        
        if not batch_X:
            # Return dummy batch with correct shape to avoid crashing the training loop
            # Shape: (0, maxlen, features)
            if len(self.X) > 0 and len(self.X[0]) > 0:
                 feat_dim = self.X[0].shape[1]
                 return np.zeros((0, self.maxlen, feat_dim)), np.zeros((0, self.maxlen, feat_dim))
            return np.array([]), np.array([])
        
        if self.task == 'autoencoder':
            # Input and output are the same (reconstruction task)
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
            # Input: sequence up to -prediction_steps
            # Output: last prediction_steps frames
            batch_X_input = []
            batch_y_output = []
            
            for seq in batch_X:
                if len(seq) > self.prediction_steps:
                    batch_X_input.append(seq[:-self.prediction_steps])
                    batch_y_output.append(seq[-self.prediction_steps:])
                else:
                    # If sequence too short, use full sequence for both
                    batch_X_input.append(seq)
                    batch_y_output.append(seq)
            
            X_padded = pad_sequences(
                batch_X_input,
                maxlen=self.maxlen - self.prediction_steps,
                dtype='float32',
                padding='post',
                truncating='post',
                value=0.0
            )
            
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
