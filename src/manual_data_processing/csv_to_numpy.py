import csv
import numpy as np
import os

class NFLDataLoader:
    """
    Loads and processes NFL Big Data Bowl 2026 data from CSV files.
    Filters input data for 'player_to_predict' == True and aligns with output data.
    """
    def __init__(self, train_dir):
        self.train_dir = train_dir
        self.input_sequences = {}
        self.output_sequences = {}
        self.input_header = []
        self.output_header = []

    def process_value(self, val):
        """
        Converts a CSV string value into the appropriate type.
        """
        val_lower = val.lower()

        # Handle Booleans
        if val_lower == 'true':
            return 1.0
        if val_lower == 'false':
            return 0.0
        
        # Handle Direction (left/right)
        if val_lower == 'left':
            return 0.0
        if val_lower == 'right':
            return 1.0

        # Handle Player Side (defense/offense)
        if val_lower == 'defense':
            return 0.0
        if val_lower == 'offense':
            return 1.0
        
        # Handle Numbers (Integers and Floats)
        try:
            return float(val)
        except ValueError:
            pass
            
        # Handle Strings (Object type)
        return str(val)

    def load_input_files(self):
        """
        Loads and filters input CSV files.
        """
        input_files = sorted([f for f in os.listdir(self.train_dir) if f.startswith('input') and f.endswith('.csv')])
        print(f"Loading and filtering {len(input_files)} Input files...")
        
        for input_file in input_files:
            input_path = os.path.join(self.train_dir, input_file)
            with open(input_path, 'r') as f:
                reader = csv.reader(f)
                first_row = True
                
                # Indices for ID columns
                player_to_predict_idx = -1
                game_id_idx = -1
                play_id_idx = -1
                nfl_id_idx = -1

                for row in reader:
                    if first_row:
                        if not self.input_header:
                            self.input_header = row
                        
                        try:
                            player_to_predict_idx = row.index('player_to_predict')
                            game_id_idx = row.index('game_id')
                            play_id_idx = row.index('play_id')
                            nfl_id_idx = row.index('nfl_id')
                        except ValueError as e:
                            print(f"Error finding columns in {input_file}: {e}")
                            break
                        
                        first_row = False
                        continue
                    
                    # Filter: Only keep rows where player_to_predict is True
                    if player_to_predict_idx != -1:
                        val = row[player_to_predict_idx].lower()
                        if val != 'true':
                            continue 

                    # Extract Key
                    key = (row[game_id_idx], row[play_id_idx], row[nfl_id_idx])
                    
                    if key not in self.input_sequences:
                        self.input_sequences[key] = []
                    self.input_sequences[key].append([self.process_value(item) for item in row])

    def load_output_files(self):
        """
        Loads output CSV files.
        """
        output_files = sorted([f for f in os.listdir(self.train_dir) if f.startswith('output') and f.endswith('.csv')])
        print(f"Loading {len(output_files)} Output files...")
        
        for output_file in output_files:
            output_path = os.path.join(self.train_dir, output_file)
            with open(output_path, 'r') as f:
                reader = csv.reader(f)
                first_row = True
                
                # Indices for ID columns
                game_id_idx = -1
                play_id_idx = -1
                nfl_id_idx = -1

                for row in reader:
                    if first_row:
                        if not self.output_header:
                            self.output_header = row
                        
                        try:
                            game_id_idx = row.index('game_id')
                            play_id_idx = row.index('play_id')
                            nfl_id_idx = row.index('nfl_id')
                        except ValueError as e:
                            print(f"Error finding columns in {output_file}: {e}")
                            break

                        first_row = False
                        continue
                    
                    # Extract Key
                    key = (row[game_id_idx], row[play_id_idx], row[nfl_id_idx])
                    
                    if key not in self.output_sequences:
                        self.output_sequences[key] = []
                    self.output_sequences[key].append([float(item) for item in row])

    def get_aligned_data(self):
        """
        Aligns input and output sequences and returns NumPy arrays.
        Returns:
            X (np.ndarray): Input sequences
            y (np.ndarray): Output sequences
        """
        self.load_input_files()
        self.load_output_files()

        print("Aligning Input and Output sequences...")
        common_keys = sorted(list(set(self.input_sequences.keys()).intersection(set(self.output_sequences.keys()))))

        aligned_X = []
        aligned_y = []

        for key in common_keys:
            aligned_X.append(self.input_sequences[key])
            aligned_y.append(self.output_sequences[key])

        print(f"Processing complete.")
        print(f"Total Unique Sequences (Matches): {len(common_keys)}")

        if not aligned_X:
            print("No matching data found.")
            return np.array([]), np.array([])

        # Convert to NumPy arrays
        # Using dtype=object to handle potential variable lengths or mixed types safely
        try:
            X = np.array(aligned_X, dtype=object)
            print(f"Initial X shape: {X.shape}")
        except Exception as e:
            print(f"Error creating X array: {e}")
            X = np.array([])

        try:
            y = np.array(aligned_y, dtype=object)
            print(f"Initial y shape: {y.shape}")
        except Exception as e:
            print(f"Error creating y array: {e}")
            y = np.array([])
            
        return X, y

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence

class NFLDataSequence(Sequence):
    """
    Keras Sequence for NFL data with automatic padding of variable-length sequences.
    """
    def __init__(self, X, y, batch_size=32, maxlen_x=None, maxlen_y=None, shuffle=True):
        """
        Args:
            X (list): List of input sequences (each sequence is a list of time steps)
            y (list): List of output sequences (each sequence is a list of time steps)
            batch_size (int): Batch size
            maxlen_x (int, optional): Maximum length for input sequences. If None, uses max length in data.
            maxlen_y (int, optional): Maximum length for output sequences. If None, uses max length in data.
            shuffle (bool): Whether to shuffle data at the end of each epoch
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
        """Number of batches per epoch"""
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Generate one batch of data
        """
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get batch data
        batch_X = [self.X[i] for i in batch_indices]
        batch_y = [self.y[i] for i in batch_indices]
        
        # Process X sequences: handle mixed types
        # The data from process_value() should already have numeric types as floats
        # and strings as strings. We need to filter out or encode string columns.
        batch_X_numeric = []
        for seq in batch_X:
            seq_numeric = []
            for frame in seq:
                frame_numeric = []
                for item in frame:
                    # If item is already a float or int (from process_value), keep it
                    if isinstance(item, (int, float)):
                        frame_numeric.append(float(item))
                    # If it's a string, we need to handle it
                    # For now, let's use a hash or skip it
                    # Better approach: filter these columns out or use proper encoding
                    elif isinstance(item, str):
                        # Try to convert to float, if fails, use hash or 0
                        try:
                            frame_numeric.append(float(item))
                        except ValueError:
                            # For non-numeric strings, use a simple hash-based encoding
                            # This is a simple placeholder - ideally use proper categorical encoding
                            frame_numeric.append(float(hash(item) % 10000))
                    else:
                        frame_numeric.append(0.0)
                seq_numeric.append(frame_numeric)
            batch_X_numeric.append(seq_numeric)
        
        # Use pad_sequences for both X and y
        # pad_sequences expects sequences of shape (n_samples, n_timesteps) for 2D
        # For 3D (n_samples, n_timesteps, n_features), we need to pad manually or use padding='post'
        
        # Method: Pad each sequence to maxlen, filling with zeros
        X_padded = pad_sequences(
            batch_X_numeric, 
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
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_tf_datasets(X, y, test_size=0.2, batch_size=32, maxlen_x=None, maxlen_y=None):
    """
    Splits X and y into training and validation sets and creates Keras Sequence datasets.
    Uses keras.utils.Sequence with padding to handle variable-length sequences.
    
    Args:
        X (np.ndarray): Input data (object array of variable-length sequences).
        y (np.ndarray): Output data (object array of variable-length sequences).
        test_size (float): Proportion of the dataset to include in the validation split.
        batch_size (int): Batch size for the datasets.
        maxlen_x (int, optional): Maximum length for input sequences. If None, auto-detects.
        maxlen_y (int, optional): Maximum length for output sequences. If None, auto-detects.
        
    Returns:
        train_sequence (NFLDataSequence): Training data sequence.
        val_sequence (NFLDataSequence): Validation data sequence.
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
