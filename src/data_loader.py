
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
    Gets the feature and label specifications from a TensorFlow Dataset.

    Args:
        dataset (tf.data.Dataset): The TensorFlow Dataset.

    Returns:
        tuple: A tuple containing the feature and label specifications.
               (feature_spec, label_spec)
    """
    element_spec = dataset.element_spec
    return element_spec[0], element_spec[1]

def create_preprocessor(features_df: pd.DataFrame):
    """
    Creates a preprocessor for the NFL Big Data Bowl 2026 prediction data.

    Args:
        features_df (pd.DataFrame): The dataframe with the features.

    Returns:
        ColumnTransformer: The preprocessor.
    """
    categorical_features = ['play_direction', 'player_position', 'player_side', 'player_role', 'nfl_id']
    numerical_features = ['x', 'y', 's', 'a', 'dir', 'o', 'absolute_yardline_number', 'player_weight', 'num_frames_output', 'ball_land_x', 'ball_land_y', 'age', 'height_inches']
    boolean_features = ['player_to_predict']

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    boolean_transformer = FunctionTransformer(lambda x: x.astype(int))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('bool', boolean_transformer, boolean_features)
        ],
        remainder='drop'
    )

    return preprocessor

def height_to_inches(height_str):
    """
    Converts height string 'feet-inches' to inches.
    """
    if isinstance(height_str, str):
        feet, inches = map(int, height_str.split('-'))
        return feet * 12 + inches
    return np.nan

SEQUENCE_LENGTH = 10

def _create_sequences_for_group(group_df: pd.DataFrame, sequence_length):
    """
    Creates sequences of features and corresponding labels for a single player/play group.
    """
    # Ensure the DataFrame is sorted by frame_id
    group_df = group_df.sort_values(by='frame_id').reset_index(drop=True)

    num_frames = len(group_df)
    if num_frames < sequence_length + 1:
        return np.array([]), np.array([])

    # Extract features and labels as numpy arrays
    feature_cols = [col for col in group_df.columns if col not in ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x_label', 'y_label']]
    features_array = group_df[feature_cols].values
    labels_array = group_df[['x_label', 'y_label']].values

    sequences = []
    labels = []

    # Number of complete sequences that can be formed
    num_sequences = num_frames - sequence_length

    for i in range(num_sequences):
        sequences.append(features_array[i : i + sequence_length])
        labels.append(labels_array[i + sequence_length])

    return np.array(sequences), np.array(labels)


def load_and_prepare_data(data_dir, test_size=0.2, random_state=42):
    """
    Loads input and output data from CSV files in the specified directory,
    merges them, preprocesses the features, splits them into training and 
    validation sets, and returns them as TensorFlow Datasets.
    The data is prepared into sequences of SEQUENCE_LENGTH frames.

    Args:
        data_dir (str): The path to the directory containing the training data.
        test_size (float): The proportion of the dataset to allocate to the validation set.
        random_state (int): The seed for the random number generator used for the split.

    Returns:
        tuple: A tuple containing the training and validation TensorFlow Datasets,
               and the preprocessor.
               (train_dataset, val_dataset, preprocessor)
    """
    input_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('input')])
    output_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('output')])

    input_dfs = [pd.read_csv(f) for f in input_files]
    output_dfs = [pd.read_csv(f) for f in output_files]

    input_df = pd.concat(input_dfs, ignore_index=True)
    output_df = pd.concat(output_dfs, ignore_index=True)

    merged_df = pd.merge(input_df, output_df, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], suffixes=('', '_label'))

    # Feature Engineering
    merged_df['height_inches'] = merged_df['player_height'].apply(height_to_inches)
    
    game_date_str = merged_df['game_id'].astype(str).str[:8]
    game_date = pd.to_datetime(game_date_str, format='%Y%m%d')
    player_birth_date = pd.to_datetime(merged_df['player_birth_date'])
    merged_df['age'] = (game_date - player_birth_date).dt.days / 365.25

    all_sequences = []
    all_labels = []

    # Define the columns that will be used as features for the preprocessor
    # This list should exclude labels and identifiers that are not model features
    feature_cols_for_model = [
        'x', 'y', 's', 'a', 'dir', 'o', 'absolute_yardline_number',
        'player_weight', 'num_frames_output', 'ball_land_x', 'ball_land_y',
        'age', 'height_inches', 'play_direction', 'player_position',
        'player_side', 'player_role', 'nfl_id', 'player_to_predict'
    ]

    # Create a DataFrame with only the features that will be preprocessed
    # This is what the preprocessor will be fitted on
    features_for_preprocessor_fitting = merged_df[feature_cols_for_model]

    preprocessor = create_preprocessor(features_for_preprocessor_fitting)
    preprocessor.fit(features_for_preprocessor_fitting) # Fit the preprocessor here

    # Apply preprocessing to the entire feature set
    # This will return a sparse matrix, convert to dense array for sequence creation
    processed_features_array = preprocessor.transform(features_for_preprocessor_fitting).toarray()
    
    # Create a DataFrame from the processed features to easily merge back with identifiers
    processed_features_df = pd.DataFrame(processed_features_array, index=merged_df.index)
    
    # Add back identifiers needed for grouping and labels
    processed_df = pd.concat([merged_df[['game_id', 'play_id', 'nfl_id', 'frame_id', 'x_label', 'y_label']], processed_features_df], axis=1)

    all_sequences = []
    all_labels = []

    # Group by game, play, and player to create sequences
    for (game_id, play_id, nfl_id), group_df in processed_df.groupby(['game_id', 'play_id', 'nfl_id']):
        sequences, labels = _create_sequences_for_group(group_df, SEQUENCE_LENGTH)
        print(f"Group: {game_id}, {play_id}, {nfl_id} - Sequences length: {len(sequences)}, Labels length: {len(labels)}")
        if sequences.size > 0 and labels.size > 0:
            all_sequences.append(sequences)
            all_labels.append(labels)

    if not all_sequences:
        raise ValueError("No sequences could be created. Please check data and SEQUENCE_LENGTH.")

    X = np.concatenate(all_sequences, axis=0)
    y = np.concatenate(all_labels, axis=0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    return train_dataset, val_dataset, preprocessor

