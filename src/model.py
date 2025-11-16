import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# These imports are needed to replicate the feature engineering from training
from data_loader import height_to_inches, SEQUENCE_LENGTH

MODEL_PATH = 'nfl_model.h5'
PREPROCESSOR_PATH = 'preprocessor.joblib'

def load_artifacts():
    """
    Loads the trained Keras model and the preprocessor from disk.
    Raises FileNotFoundError if either artifact is missing.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first by running predictor.py.")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}. Please train the model first.")

    print(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"Loading preprocessor from {PREPROCESSOR_PATH}")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    return model, preprocessor

# Load the model globally to avoid reloading it for each batch.
model, preprocessor = load_artifacts()

def preprocess_features(test_df, test_input_df):
    """
    Preprocesses the raw input dataframes into a format the model expects.
    This function replicates the feature engineering and sequence creation from
    the training pipeline (`data_loader.py`).
    
    Args:
        test_df (pd.DataFrame): The dataframe with the rows to predict.
        test_input_df (pd.DataFrame): The dataframe with the input features for the play.

    Returns:
        np.array: A 3D array of shape (num_predictions, SEQUENCE_LENGTH, num_features)
                  ready to be fed into the LSTM model.
    """
    num_predictions = len(test_df)
    if num_predictions == 0:
        return np.array([])

    # Combine input data for the entire play. `test_input_df` contains frame 0 (the context),
    # and `test_df` contains the frames we need to predict for.
    play_df = pd.concat([test_input_df, test_df], ignore_index=True)
    play_df = play_df.sort_values(by=['nfl_id', 'frame_id']).reset_index(drop=True)

    # 1. Recreate the exact same features as in training
    play_df['height_inches'] = play_df['player_height'].apply(height_to_inches)
    game_date_str = play_df['game_id'].astype(str).str[:8]
    game_date = pd.to_datetime(game_date_str, format='%Y%m%d')
    player_birth_date = pd.to_datetime(play_df['player_birth_date'])
    play_df['age'] = (game_date - player_birth_date).dt.days / 365.25

    # 2. Apply the pre-fitted preprocessor
    feature_cols = preprocessor.feature_names_in_
    processed_features_array = preprocessor.transform(play_df[feature_cols])
    processed_features_df = pd.DataFrame(processed_features_array, index=play_df.index)

    # Add back identifiers needed for sequence creation
    processed_df = pd.concat([play_df[['nfl_id', 'frame_id']], processed_features_df], axis=1)
    
    # 3. Create sequences for each row in the original `test_df` (each row to predict)
    sequences = []
    for _, row_to_predict in test_df.iterrows():
        player_id = row_to_predict['nfl_id']
        frame_id = row_to_predict['frame_id']
        
        # Find the player's data and the exact frame we need to predict
        player_data = processed_df[processed_df['nfl_id'] == player_id]
        prediction_frame_index = player_data[player_data['frame_id'] == frame_id].index[0]
        
        # The sequence consists of the `SEQUENCE_LENGTH` frames *before* the prediction frame
        start_idx = prediction_frame_index - SEQUENCE_LENGTH
        end_idx = prediction_frame_index
        
        sequence = player_data.iloc[start_idx:end_idx].drop(columns=['nfl_id', 'frame_id']).values
        sequences.append(sequence)

    return np.array(sequences)

def predict(test_df, test_input_df):
    """
    Generates predictions for a single batch (play).
    """
    # The gateway provides polars dataframes, convert them to pandas
    test_df = test_df.to_pandas()
    test_input_df = test_input_df.to_pandas()

    # 1. Preprocess the data to create features for the model
    features = preprocess_features(test_df, test_input_df)

    if features.shape[0] == 0:
        return pd.DataFrame([], columns=['x', 'y'])

    # 2. Run inference
    # Calling the model directly is often faster for inference than model.predict()
    predictions_xy = model(features, training=False).numpy()

    # 3. Format the predictions into the required DataFrame
    return pd.DataFrame(predictions_xy, columns=['x', 'y'])