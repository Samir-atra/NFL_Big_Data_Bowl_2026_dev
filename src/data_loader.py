import os
# Force JAX to use CPU due to GPU conflicts and memory issues in the current environment.
os.environ["JAX_PLATFORM_NAME"] = "cpu"
# Force TensorFlow to use CPU due to GPU conflicts and memory issues in the current environment.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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

def create_preprocessor(features_df):
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
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
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
    """Converts height string 'feet-inches' to inches."""
    if isinstance(height_str, str):
        feet, inches = map(int, height_str.split('-'))
        return feet * 12 + inches
    return jnp.nan

SEQUENCE_LENGTH = 10

def create_sequences(df, sequence_length, feature_columns_for_model):
    """
    Creates sequences of features and corresponding labels for a single player/play.
    Each sequence consists of `sequence_length` frames, and the label is the
    x_label, y_label of the frame immediately following the sequence.
    """
    sequences = []
    labels = []
    # Ensure the DataFrame is sorted by frame_id
    df = df.sort_values(by='frame_id').reset_index(drop=True)

    for i in range(len(df) - sequence_length):
        # Features are frames i to i + sequence_length - 1, selecting only model features
        seq_features = df.iloc[i:i + sequence_length][feature_columns_for_model]
        # Label is frame i + sequence_length
        seq_label = df.iloc[i + sequence_length][['x_label', 'y_label']]
        
        sequences.append(seq_features)
        labels.append(seq_label)
    
    return sequences, labels

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

    # Group by game, play, and player to create sequences
    for (game_id, play_id, nfl_id), group_df in merged_df.groupby(['game_id', 'play_id', 'nfl_id']):
        if len(group_df) >= SEQUENCE_LENGTH + 1: # Need at least SEQUENCE_LENGTH + 1 frames for one sequence and its label
            sequences, labels = create_sequences(group_df, SEQUENCE_LENGTH, feature_cols_for_model) # Pass feature_cols_for_model
            all_sequences.extend(sequences)
            all_labels.extend(labels)

    if not all_sequences:
        raise ValueError("No sequences could be created. Check data and SEQUENCE_LENGTH.")

    # Transform all sequences using the already fitted preprocessor
    processed_sequences = []
    for seq_df in all_sequences:
        # Ensure seq_df only contains the columns the preprocessor was fitted on
        processed_seq = preprocessor.transform(seq_df).toarray()
        processed_sequences.append(processed_seq)

    X = jnp.array(processed_sequences)
    y = jnp.array(pd.DataFrame(all_labels).values)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    return train_dataset, val_dataset, preprocessor

if __name__ == '__main__':
    data_directory = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train'
    try:
        train_ds, val_ds, preprocessor = load_and_prepare_data(data_directory)
        print("Successfully created training and validation datasets.")
        print("Training dataset:", train_ds)
        print("Validation dataset:", val_ds)

        feature_spec, label_spec = get_feature_label_specs(train_ds)
        print("\n--- Feature Spec ---")
        print(feature_spec)
        print("\n--- Label Spec ---")
        print(label_spec)
        
        # Example of how to inspect the first element
        for features, label in train_ds.take(1):
            print("\n--- Example processed feature batch ---")
            print(features.numpy())
            print("\n--- Example label ---")
            print(label.numpy())

    except FileNotFoundError:
        print(f"Error: The directory '{data_directory}' was not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
