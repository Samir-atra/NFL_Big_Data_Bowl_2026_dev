"""
Module for loading pre-trained models and preprocessing data for inference.

This module is responsible for loading the trained Keras model and the fitted
scikit-learn preprocessor used during training. It then provides functions to:

1.  **Load Artifacts (`load_artifacts`):** Loads the model (`.h5` or `.keras` file)
    and the preprocessor (`.joblib` file) from disk.
2.  **Preprocess Features (`preprocess_features`):** Takes raw player data for a
    play (context and target frames) and transforms it into the exact sequence
    format expected by the trained model. This involves replicating feature
    engineering steps (like calculating age, converting height) and applying
    the fitted scikit-learn preprocessor.
3.  **Predict (`predict`):** Orchestrates the prediction process for a given batch
    of data. It calls `preprocess_features` to prepare the input and then uses
    the loaded Keras model for inference, returning the predictions in a
    Pandas DataFrame.

This module is crucial for the inference pipeline, enabling the application
of the trained model to new, unseen data.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# These imports are needed to replicate the feature engineering from training.
# Specifically, `height_to_inches` for feature transformation and `SEQUENCE_LENGTH`
# for creating input sequences of the correct temporal dimension.
from data_loader import height_to_inches, SEQUENCE_LENGTH

# --- Artifact Paths ---
# Define the file paths for the trained model and the preprocessor.
# These should correspond to where these artifacts are saved after training.
MODEL_PATH = 'nfl_model.keras' # Updated to .keras for newer TensorFlow versions, assuming it's compatible.
PREPROCESSOR_PATH = 'preprocessor.joblib'

def load_artifacts() -> tuple[tf.keras.Model, object]:
    """
    Loads the trained Keras model and the pre-fitted scikit-learn preprocessor from disk.

    This function is critical for inference, as it loads the necessary components
    that were saved after the model training phase.

    Returns:
        tuple[tf.keras.Model, object]: A tuple containing:
            - The loaded Keras model.
            - The loaded scikit-learn preprocessor object.

    Raises:
        FileNotFoundError:
            If the model file (`MODEL_PATH`) or the preprocessor file (`PREPROCESSOR_PATH`)
            does not exist at the specified locations.
    """
    # Check if the model file exists.
    if not os.path.exists(MODEL_PATH):
        # Raise an error if the model file is not found, guiding the user to train the model first.
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first by running predictor.py.")
    # Check if the preprocessor file exists.
    if not os.path.exists(PREPROCESSOR_PATH):
        # Raise an error if the preprocessor file is not found, guiding the user to train the model first.
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}. Please train the model first.")

    # Load the Keras model from the specified path.
    print(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load the scikit-learn preprocessor (e.g., ColumnTransformer) from the joblib file.
    print(f"Loading preprocessor from {PREPROCESSOR_PATH}")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # Return both the loaded model and preprocessor.
    return model, preprocessor

# --- Global Artifact Loading ---
# Load the model and preprocessor globally when the module is imported.
# This avoids the overhead of reloading them every time a prediction is made,
# making inference faster, especially for batch predictions.
try:
    model, preprocessor = load_artifacts()
except FileNotFoundError as e:
    print(f"Error loading model artifacts: {e}")
    # Depending on the application's design, you might want to exit or handle this.
    # For now, we let it potentially cause errors later if used without artifacts.
    model, preprocessor = None, None # Explicitly set to None if loading fails

def preprocess_features(test_df: pd.DataFrame, test_input_df: pd.DataFrame) -> np.ndarray:
    """
    Preprocesses raw input dataframes into a format compatible with the trained model.

    This function replicates the exact feature engineering steps and preprocessing pipeline
    used during the model training phase (as defined in `data_loader.py` and the
    preprocessor fitted during training).
    It combines context frames (`test_input_df`) with target frames (`test_df`),
    recalculates necessary features (like age, height in inches), applies the pre-fitted
    scikit-learn preprocessor, and then structures the data into sequences of the
    correct length (`SEQUENCE_LENGTH`) and shape for model inference.

    Args:
        test_df (pd.DataFrame): DataFrame containing the target rows (frames) for which
                                predictions are needed. Each row represents a specific
                                player at a specific frame in a play.
        test_input_df (pd.DataFrame): DataFrame containing the contextual information for
                                      the play, typically including frame 0 and potentially
                                      frames preceding `test_df` if they are part of the
                                      sequence context.

    Returns:
        np.ndarray: A 3D NumPy array of shape `(num_predictions, SEQUENCE_LENGTH, num_features)`
                    ready to be fed into the LSTM model for inference. Returns an empty
                    array if no predictions can be made (e.g., `test_df` is empty).
    """
    # Get the number of prediction targets (rows in test_df).
    num_predictions = len(test_df)
    # If there are no prediction targets, return an empty array immediately.
    if num_predictions == 0:
        return np.array([])

    # Combine the input context data with the target prediction data for the entire play.
    # `test_input_df` usually contains the historical context (e.g., frame 0),
    # and `test_df` contains the specific frames for which we need to predict player positions.
    play_df = pd.concat([test_input_df, test_df], ignore_index=True)
    # Sort the combined DataFrame by player ID (`nfl_id`) and frame ID (`frame_id`) 
    # to ensure correct temporal ordering within sequences.
    play_df = play_df.sort_values(by=['nfl_id', 'frame_id']).reset_index(drop=True)

    # --- Feature Engineering (Replication of Training Steps) ---
    # Recreate features that were generated during training. This must exactly match the training process.
    
    # Convert player height from string format (e.g., '6-1') to total inches.
    play_df['height_inches'] = play_df['player_height'].apply(height_to_inches)
    
    # Calculate player age. Extract the date from `game_id` (assumed YYYYMMDD format)
    # and subtract the player's birth date.
    game_date_str = play_df['game_id'].astype(str).str[:8] # Extract YYYYMMDD from game_id
    game_date = pd.to_datetime(game_date_str, format='%Y%m%d')
    player_birth_date = pd.to_datetime(play_df['player_birth_date'])
    play_df['age'] = (game_date - player_birth_date).dt.days / 365.25 # Age in years

    # --- Apply Pre-fitted Preprocessor ---
    # Get the list of feature column names that the preprocessor was fitted on.
    # This ensures we only transform the relevant columns and in the correct order.
    feature_cols = preprocessor.feature_names_in_
    # Apply the transformation using the pre-fitted preprocessor.
    processed_features_array = preprocessor.transform(play_df[feature_cols])
    # Convert the transformed array back to a DataFrame for easier manipulation.
    processed_features_df = pd.DataFrame(processed_features_array, index=play_df.index)

    # Add back essential identifiers (`nfl_id`, `frame_id`) needed for sequence creation,
    # along with the processed features.
    processed_df = pd.concat([play_df[['nfl_id', 'frame_id']], processed_features_df], axis=1)
    
    # --- Sequence Creation ---
    # Create input sequences for each prediction target row in the original `test_df`.
    sequences = []
    # Iterate through each row in `test_df` which represents a specific frame
    # for which we need to predict the player's (x, y) coordinates.
    for _, row_to_predict in test_df.iterrows():
        player_id = row_to_predict['nfl_id']
        frame_id = row_to_predict['frame_id']
        
        # Find all data for the current player within the processed DataFrame.
        player_data = processed_df[processed_df['nfl_id'] == player_id]
        # Find the index in `player_data` corresponding to the exact frame we need to predict.
        # `.index[0]` is used because `player_data['frame_id'] == frame_id` might return multiple indices if frame_id is not unique per player (unlikely but safer).
        prediction_frame_index = player_data[player_data['frame_id'] == frame_id].index[0]
        
        # Define the start and end indices for the input sequence.
        # The sequence consists of `SEQUENCE_LENGTH` frames *preceding* the prediction frame.
        start_idx = prediction_frame_index - SEQUENCE_LENGTH
        end_idx = prediction_frame_index # The frame *before* this index is the last frame in the sequence.
        
        # Extract the sequence of features for this player, from `start_idx` up to (but not including) `end_idx`.
        # Drop the identifier columns (`nfl_id`, `frame_id`) as they are not model features.
        sequence = player_data.iloc[start_idx:end_idx].drop(columns=['nfl_id', 'frame_id']).values
        sequences.append(sequence)

    # Convert the list of sequences into a NumPy array. This array will have the shape:
    # (num_predictions, SEQUENCE_LENGTH, num_features).
    return np.array(sequences)

def predict(test_df: pd.DataFrame, test_input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates predictions for a single batch of data (representing one or more plays).

    This function takes raw dataframes containing context and target information,
    preprocesses them into the format required by the model, runs inference using
    the loaded Keras model, and returns the predicted (x, y) coordinates.

    Args:
        test_df (pd.DataFrame): DataFrame containing the target rows (frames) for prediction.
                                This typically comes from the gateway's `test_df`.
        test_input_df (pd.DataFrame): DataFrame containing the contextual input frames for the play.
                                      This typically comes from the gateway's `test_input_df`.

    Returns:
        pd.DataFrame: A DataFrame with two columns, 'x' and 'y', representing the predicted
                      coordinates for each prediction target. Returns an empty DataFrame if
                      no predictions are made.
    """
    # The gateway often provides data in Polars DataFrames. Convert them to Pandas DataFrames
    # as the rest of this module expects and works with Pandas.
    test_df = test_df.to_pandas()
    test_input_df = test_input_df.to_pandas()

    # 1. Preprocess the data: Convert raw data into sequences of features that the model can understand.
    # This step is crucial and must mirror the preprocessing done during training.
    features = preprocess_features(test_df, test_input_df)

    # If preprocessing results in no features (e.g., empty input data), return an empty DataFrame.
    if features.shape[0] == 0:
        return pd.DataFrame([], columns=['x', 'y'])

    # 2. Run inference using the loaded Keras model.
    # `model(features, training=False)` is generally preferred for inference over `model.predict(features)`
    # as it can be slightly faster and more explicit about disabling training-specific behavior (like dropout).
    # `.numpy()` converts the TensorFlow tensor output to a NumPy array.
    predictions_xy = model(features, training=False).numpy()

    # 3. Format the predictions into the required output structure.
    # The model is expected to output (x, y) coordinates, so we create a Pandas DataFrame
    # with columns 'x' and 'y' for the predictions.
    return pd.DataFrame(predictions_xy, columns=['x', 'y'])