
"""
Module for loading, preprocessing, and preparing NFL Big Data Bowl data for machine learning models.

This module provides functionalities to:
1. Load raw CSV data for input features and output labels.
2. Perform feature engineering, including converting height strings to inches and calculating player age.
3. Create and fit a scikit-learn `ColumnTransformer` preprocessor to handle numerical, categorical, and boolean features.
4. Generate fixed-length sequences of features and corresponding labels for training.
5. Split the data into training and validation sets.
6. Convert the prepared data into TensorFlow Datasets for efficient model training.
"""
import os
import pandas as pd
import tensorflow as tf
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def get_feature_label_specs(dataset):
    """
    Retrieves the element specification (feature and label shapes and dtypes)
    from a TensorFlow Dataset.

    This is useful for understanding the expected input/output format of the dataset,
    especially when building Keras models or debugging data pipelines.

    Args:
        dataset (tf.data.Dataset): The TensorFlow Dataset object.

    Returns:
        tuple: A tuple containing two elements: the feature specification and the 
               label specification. Each specification describes the shape and dtype
               of the corresponding tensor in the dataset elements.
    """
    # Get the element_spec from the dataset, which describes the structure
    # of each element (e.g., a tuple of (features, labels))
    element_spec = dataset.element_spec
    # Return the feature spec (element_spec[0]) and label spec (element_spec[1])
    return element_spec[0], element_spec[1]

def create_preprocessor(features_df: pd.DataFrame):
    """
    Creates a scikit-learn ColumnTransformer preprocessor tailored for the NFL Big Data Bowl data.

    This preprocessor handles different feature types:
    - Numerical features: Scaled using StandardScaler.
    - Categorical features: One-hot encoded using OneHotEncoder.
    - Boolean features: Converted to integers (0 or 1).

    All other columns (like identifiers) are dropped.

    Args:
        features_df (pd.DataFrame): A sample DataFrame containing the feature columns
                                     to define the preprocessing steps.

    Returns:
        ColumnTransformer: The configured scikit-learn preprocessor object.
    """
    # Define feature types based on domain knowledge and common preprocessing needs
    categorical_features = ['play_direction', 'player_position', 'player_side', 'player_role', 'nfl_id']
    numerical_features = ['x', 'y', 's', 'a', 'dir', 'o', 'absolute_yardline_number', 'player_weight', 'num_frames_output', 'ball_land_x', 'ball_land_y', 'age', 'height_inches']
    boolean_features = ['player_to_predict']

    # Define transformers for each feature type
    numerical_transformer = StandardScaler() # Scales numerical features to have zero mean and unit variance
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Encodes categorical features into one-hot vectors
    boolean_transformer = FunctionTransformer(lambda x: x.astype(int)) # Converts boolean features (True/False) to integers (1/0)

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('bool', boolean_transformer, boolean_features)
        ],
        remainder='drop' # Drop any columns not explicitly mentioned in transformers
    )

    return preprocessor

def height_to_inches(height_str):
    """
    Converts a height string in the format 'feet-inches' to the total number of inches.

    For example, '6-1' would be converted to 73 inches.
    Handles potential non-string or malformed inputs gracefully by returning NaN.

    Args:
        height_str (str): The height string (e.g., '6-1').

    Returns:
        float or None: The height in inches, or None if the input is invalid.
    """
    if isinstance(height_str, str):
        feet, inches = map(int, height_str.split('-'))
        return feet * 12 + inches
    return np.nan

SEQUENCE_LENGTH = 10

def _create_sequences_for_group(group_df: pd.DataFrame, sequence_length):
    """
    Creates input feature sequences and corresponding output labels from a grouped DataFrame.
    This function takes a DataFrame representing a single play for a single player,
    sorts it by `frame_id`, and then slides a window of `sequence_length` to create
    input sequences and extracts the label from the frame immediately following the sequence.

    Args:
        group_df (pd.DataFrame): DataFrame for a single player's data within a play, 
                                 already grouped and filtered.
        sequence_length (int): The fixed number of frames to include in each input sequence.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - sequences (np.ndarray): Array of shape (n_sequences, sequence_length, n_features).
            - labels (np.ndarray): Array of shape (n_sequences, n_label_features).
              Returns empty arrays if not enough frames are available to form sequences.
    """

    # Ensure the DataFrame is sorted by frame_id to maintain temporal order within sequences
    group_df = group_df.sort_values(by='frame_id').reset_index(drop=True)

    num_frames = len(group_df)
    # We need at least sequence_length frames for input + 1 frame for the label
    if num_frames < sequence_length + 1:
        return np.array([]), np.array([])

    # Extract features and labels as numpy arrays
    # Define feature columns, excluding identifiers and labels
    feature_cols = [col for col in group_df.columns if col not in ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x_label', 'y_label']]
    features_array = group_df[feature_cols].values
    labels_array = group_df[['x_label', 'y_label']].values

    sequences = []
    labels = []

    # Create sliding windows for sequences and labels
    # Number of complete sequences that can be formed from the available frames
    num_sequences = num_frames - sequence_length

    for i in range(num_sequences):
        # Append the sequence of frames from i to i + sequence_length
        sequences.append(features_array[i : i + sequence_length])
        # Append the label from the frame immediately following the sequence
        labels.append(labels_array[i + sequence_length])

    return np.array(sequences), np.array(labels)


def load_and_prepare_data(data_dir, test_size=0.2, random_state=42):
    """
    Loads input and output data from CSV files in the specified directory,
    merges them based on common identifiers, preprocesses features, 
    splits the data into training and validation sets, and converts them into 
    TensorFlow Datasets.

    The data is transformed into fixed-length sequences of `SEQUENCE_LENGTH` frames.

    Args:
        data_dir (str): The path to the directory containing the training data CSV files.
        test_size (float, optional): The proportion of the dataset to include in the 
            validation split. Defaults to 0.2.
        random_state (int, optional): The seed for random number generation to ensure 
            reproducibility of the train-validation split. Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - train_dataset (tf.data.Dataset): TensorFlow Dataset for training.
            - val_dataset (tf.data.Dataset): TensorFlow Dataset for validation.
            - preprocessor (ColumnTransformer): The fitted scikit-learn preprocessor.
            
    Raises:
        ValueError: If no sequences can be created (e.g., due to insufficient data 
                    or incorrect SEQUENCE_LENGTH).
    """
    # Find and sort input and output CSV files
    input_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('input')])
    output_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('output')])

    # Load all input and output CSVs into Pandas DataFrames
    input_dfs = [pd.read_csv(f) for f in input_files]
    output_dfs = [pd.read_csv(f) for f in output_files]

    # Concatenate all dataframes to create a single DataFrame for inputs and outputs
    input_df = pd.concat(input_dfs, ignore_index=True)
    output_df = pd.concat(output_dfs, ignore_index=True)

    # Merge input and output dataframes based on common identifiers (game, play, frame, player)
    # Suffixes are added to disambiguate columns with the same name (e.g., 'x' in both input and output)
    merged_df = pd.merge(input_df, output_df, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], suffixes=('', '_label'))

    # --- Feature Engineering ---
    # Convert player height from 'feet-inches' string to total inches
    merged_df['height_inches'] = merged_df['player_height'].apply(height_to_inches)
    
    # Calculate player age based on game date and birth date
    # Assuming game date can be derived from game_id (YYYYMMDD format)
    game_date_str = merged_df['game_id'].astype(str).str[:8]
    game_date = pd.to_datetime(game_date_str, format='%Y%m%d')
    player_birth_date = pd.to_datetime(merged_df['player_birth_date'])
    merged_df['age'] = (game_date - player_birth_date).dt.days / 365.25

    all_sequences = []
    all_labels = []

    # Define the columns that will be used as features for the preprocessor.
    # This list is crucial as it dictates which columns are included in the model's input.
    # It must exclude identifiers and labels that are not intended as model features.
    # Importantly, this list must precisely match the features used during the training phase
    # to ensure consistency between training and inference.
    feature_cols_for_model = [
        'x', 'y', 's', 'a', 'dir', 'o', 'absolute_yardline_number',
        'player_weight', 'num_frames_output', 'ball_land_x', 'ball_land_y',
        'age', 'height_inches', 'play_direction', 'player_position',
        'player_side', 'player_role', 'nfl_id', 'player_to_predict'
    ]

    # Create a DataFrame containing only the features that will be preprocessed.
    # This DataFrame is used specifically for fitting the scikit-learn preprocessor.
    features_for_preprocessor_fitting = merged_df[feature_cols_for_model]

    # Initialize and fit the preprocessor
    preprocessor = create_preprocessor(features_for_preprocessor_fitting)
    preprocessor.fit(features_for_preprocessor_fitting) # Fit the preprocessor on the feature data

    # Apply preprocessing to the entire feature set.
    # The result is typically a sparse matrix, which we convert to a dense array for sequence creation.
    processed_features_array = preprocessor.transform(features_for_preprocessor_fitting).toarray()
    
    # Create a DataFrame from the processed features. This is necessary to merge back the identifiers
    # and labels required for grouping plays and creating sequences.
    processed_features_df = pd.DataFrame(processed_features_array, index=merged_df.index)
    
    # Add back identifiers and labels to the processed features DataFrame
    processed_df = pd.concat([merged_df[['game_id', 'play_id', 'nfl_id', 'frame_id', 'x_label', 'y_label']], processed_features_df], axis=1)

    all_sequences = []
    all_labels = []

    # Group data by game, play, and player to form sequences
    # This ensures that each sequence corresponds to a single player's movement within a single play.
    for (game_id, play_id, nfl_id), group_df in processed_df.groupby(['game_id', 'play_id', 'nfl_id']):
        # Create feature sequences and corresponding labels for the current group
        sequences, labels = _create_sequences_for_group(group_df, SEQUENCE_LENGTH)
        print(f"Group: {game_id}, {play_id}, {nfl_id} - Sequences length: {len(sequences)}, Labels length: {len(labels)}")
        
        # Append only if valid sequences and labels were created
        if sequences.size > 0 and labels.size > 0:
            all_sequences.append(sequences)
            all_labels.append(labels)

    # Raise an error if no sequences could be created (e.g., insufficient data or incorrect SEQUENCE_LENGTH)
    if not all_sequences:
        raise ValueError("No sequences could be created. Please check data and SEQUENCE_LENGTH.")

    # Concatenate all collected sequences and labels into single NumPy arrays
    X = np.concatenate(all_sequences, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert the split data into TensorFlow Datasets for efficient batching and prefetching
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    return train_dataset, val_dataset, preprocessor
