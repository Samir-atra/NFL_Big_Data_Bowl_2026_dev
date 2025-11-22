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

def create_tf_datasets(X, y, test_size=0.2, batch_size=32):
    """
    Splits X and y into training and validation sets and creates tf.data.Datasets.
    
    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Output data.
        test_size (float): Proportion of the dataset to include in the validation split.
        batch_size (int): Batch size for the datasets.
        
    Returns:
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
    """
    print("\n--- Creating TensorFlow Datasets ---")
    
    # Ensure data is in a format compatible with tf.data.Dataset
    # X and y are currently object arrays of lists (sequences).
    # We need to handle the variable length or convert to a uniform tensor if possible.
    # For now, we will cast to string to handle mixed types in X, 
    # and assume y is numeric but maybe variable length.
    # Note: tf.data.Dataset.from_tensor_slices requires uniform shape for standard tensors,
    # or a ragged tensor if shapes vary.
    
    # Attempt to convert to RaggedTensors if shapes are variable, or standard tensors if uniform.
    # Since X is (N,) object array of lists, we can try to convert to RaggedTensor.
    
    try:
        # Convert X to RaggedTensor (handling variable lengths if any, and mixed types as strings)
        # Note: converting mixed type list-of-lists to tensor is tricky.
        # We'll convert everything to string for X to be safe.
        
        # Helper to convert object array of lists to list of lists for RaggedTensor
        X_list = X.tolist()
        y_list = y.tolist()
        
        # Split first
        print(f"Splitting data (test_size={test_size})...")
        X_train, X_val, y_train, y_val = train_test_split(X_list, y_list, test_size=test_size, random_state=42)
        
        print(f"Train size: {len(X_train)}")
        print(f"Val size: {len(X_val)}")
        
        # Create Datasets
        # Using from_generator is often safer for variable length/complex structures than from_tensor_slices
        # But let's try ragged constant if possible, or just generator.
        
        def generator(data_X, data_y):
            for x_seq, y_seq in zip(data_X, data_y):
                # Explicitly convert all elements to string to avoid mixed type error
                # x_seq is a list of frames (lists), so we iterate through frames and items
                x_seq_str = [[str(item) for item in frame] for frame in x_seq]
                
                # Convert y_seq to float32 tensor
                yield tf.constant(x_seq_str, dtype=tf.string), tf.constant(y_seq, dtype=tf.float32)

        output_signature = (
            tf.TensorSpec(shape=(None, None), dtype=tf.string), # (Time, Features) - variable time
            tf.TensorSpec(shape=(None, None), dtype=tf.float32) # (Time, Features) - variable time
        )

        print("Building Train Dataset...")
        train_dataset = tf.data.Dataset.from_generator(
            lambda: generator(X_train, y_train),
            output_signature=output_signature
        )
        train_dataset = train_dataset.batch(batch_size)
        
        print("Building Validation Dataset...")
        val_dataset = tf.data.Dataset.from_generator(
            lambda: generator(X_val, y_val),
            output_signature=output_signature
        )
        val_dataset = val_dataset.batch(batch_size)
        
        print("Datasets created successfully.")
        return train_dataset, val_dataset

    except Exception as e:
        print(f"Error creating TF datasets: {e}")
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

    # Create TF Datasets
    train_ds, val_ds = create_tf_datasets(X, y)
    
    if train_ds:
        print("\nVerifying Dataset Element:")
        for x_batch, y_batch in train_ds.take(1):
            print(f"Batch X shape: {x_batch.shape}")
            print(f"Batch y shape: {y_batch.shape}")

    print("\nData loading, alignment, and dataset creation complete.")
