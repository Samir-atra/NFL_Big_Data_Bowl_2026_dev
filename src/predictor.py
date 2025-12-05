"""
This module is responsible for building, training, and saving a Keras sequence-to-sequence (seq2seq) model for predicting NFL game outcomes.

It leverages custom data loading and preprocessing utilities from `csv_to_numpy` and defines an LSTM-based architecture suitable for time-series data.

Key functionalities:
- **Model Architecture Definition (`build_seq2seq_model`):** Creates a Keras model with LSTM layers designed for seq2seq tasks, including fixed and tunable parameters.
- **Model Training (`train_model`):** Trains the defined Keras model using TensorFlow datasets, incorporating callbacks for early stopping and model checkpointing.
- **Main Execution (`main`):** Orchestrates the entire process: loading data, preparing training/validation sequences, building the model, training it, and saving the final trained model.

This script is intended to be run directly to train the predictor model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys

# --- Path Configuration ---
# Add the manual_data_processing directory to the Python path to allow importing custom modules.
# This is necessary because 'csv_to_numpy.py' and related utilities are located in a subdirectory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'manual_data_processing'))

# Import necessary data handling utilities.
from csv_to_numpy import NFLDataLoader, create_tf_datasets


def build_seq2seq_model(
    input_seq_length: int,
    input_features: int,
    output_seq_length: int,
    output_features: int,
    lstm_units: int = 128,
) -> keras.Model:
    """
    Builds and compiles a Keras sequence-to-sequence (seq2seq) LSTM model.

    This function defines a deep LSTM architecture suitable for predicting sequences
    of game outcomes based on historical play data. It includes an encoder-decoder structure
    with TimeDistributed dense layers for output.

    Args:
        input_seq_length (int): The fixed length of input sequences (number of timesteps).
        input_features (int): The number of features per timestep in the input data.
        output_seq_length (int): The desired length of the output sequences (number of timesteps).
        output_features (int): The number of output features per timestep (e.g., x and y coordinates).
        lstm_units (int): The base number of units for the LSTM layers. This parameter is 
                          currently fixed in the model architecture but can be made tunable.

    Returns:
        keras.Model: A compiled Keras model ready for training.
    """

    SEED = 42 # Fixed random seed for reproducibility in model initialization.
    
    # Define the model architecture using Keras Sequential API.
    model = keras.Sequential([
        # --- Input Layer ---
        # Specifies the shape of the input data: (timesteps, features).
        layers.Input(shape=(input_seq_length, input_features)),
        
        # --- Encoder LSTM Layers ---
        # These layers process the input sequence. `return_sequences=True` is essential 
        # for stacked LSTMs or when the output is needed at each timestep.
        keras.layers.LSTM(
            units=123, # Fixed number of units for the first LSTM layer.
            activation="sigmoid", # Fixed activation function.
            return_sequences=True,
            # kernel_regularizer=keras.regularizers.L2(l2=0.00000195), # Commented out L2 regularization.
            seed=SEED,
        ),
        keras.layers.LSTM(
            units=64,
            activation="sigmoid", # Fixed activation function.
            return_sequences=True,
            # kernel_regularizer=keras.regularizers.L2(l2=kernel_r), # Commented out tunable regularization.
            seed=SEED,
        ),
        keras.layers.LSTM(
            units=64, 
            activation="sigmoid", # Fixed activation function.
            return_sequences=True,
            # kernel_regularizer=keras.regularizers.L2(l2=kernel_r), # Commented out tunable regularization.
            seed=SEED,
        ),
        keras.layers.LSTM(
            units=32,
            activation="sigmoid", # Fixed activation function.
            return_sequences=True,
            # kernel_regularizer=keras.regularizers.L2(l2=0.00000195), # Commented out L2 regularization.
            seed=SEED,
        ),
        
        # --- Output Shaping Layer ---
        # Lambda layer to crop or slice the output sequence if the LSTM's output length 
        # (which defaults to input length if return_sequences=True) differs from the 
        # desired `output_seq_length`. This ensures the output matches the target sequence length.
        layers.Lambda(lambda x: x[:, :output_seq_length, :]),
        
        # --- TimeDistributed Dense Layer ---
        # Applies a Dense layer to each timestep of the input. This is crucial for seq2seq models
        # where we want to predict a set of features for each output timestep.
        # Uses linear activation for regression tasks (predicting coordinates).
        layers.TimeDistributed(keras.layers.Dense(units=output_features, activation="linear")),
    ])

    # --- Model Compilation ---
    # Compiles the model with an optimizer, loss function, and metrics.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001), # Adam optimizer with a fixed learning rate.
        loss='mse',                                          # Mean Squared Error loss for regression.
        metrics=['mae']                                       # Mean Absolute Error as a performance metric.
    )
    
    return model

def train_model(
    model: keras.Model,
    train_sequence: tf.data.Dataset,
    val_sequence: tf.data.Dataset,
    epochs: int = 10,
    callbacks: list | None = None,
) -> keras.callbacks.History:
    """
    Trains the Keras model using provided training and validation sequences.

    This function configures and applies essential Keras callbacks, such as
    EarlyStopping to prevent overfitting and ModelCheckpoint to save the best
    performing model during training.

    Args:
        model (keras.Model): The Keras model instance to be trained.
        train_sequence (tf.data.Dataset): The dataset object for training data.
        val_sequence (tf.data.Dataset): The dataset object for validation data.
        epochs (int): The total number of epochs to train for.
        callbacks (list | None, optional): A list of additional Keras callbacks to use.
                                           Defaults to None.
    
    Returns:
        keras.callbacks.History: A History object containing records of training
                               loss values and metrics for each epoch.
    """
    if callbacks is None:
        callbacks = []
    
    # --- Callbacks Configuration ---
    # EarlyStopping: Monitors validation loss and stops training if it doesn't improve
    # for a specified number of epochs (`patience`). Restores the best weights found.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',      # Metric to monitor
        patience=5,              # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True, # Whether to restore model weights from the epoch with the best value of the monitored quantity
        verbose=1                # Print messages when training stops
    )
    
    # ModelCheckpoint: Saves the model's weights (or the entire model) at regular intervals,
    # based on a monitored quantity. `save_best_only=True` ensures only the best model is saved.
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        'best_model.keras',      # Path to save the best model
        monitor='val_loss',      # Metric to monitor for saving best model
        save_best_only=True,     # If True, only the best model is saved
        verbose=1                # Print messages when a new best model is saved
    )
    
    # Extend the provided callbacks list with the configured callbacks.
    callbacks.extend([early_stopping, model_checkpoint])
    
    print("Starting model training...")
    # --- Model Training ---
    # The `fit` method trains the model for a fixed number of epochs.
    history = model.fit(
        train_sequence,
        epochs=epochs,
        validation_data=val_sequence,
        callbacks=callbacks,
        verbose=1 # Controls the verbosity of training logs (0=silent, 1=progress bar, 2=one line per epoch)
    )
    print("Model training finished.")
    return history

def main():
    """
    Main function to orchestrate the NFL predictor model training process.

    This function performs the following steps:
    1. Sets up configuration parameters (data directory, batch size, epochs, test size).
    2. Loads and prepares the dataset using `NFLDataLoader` and `create_tf_datasets`.
    3. Determines input/output sequence and feature shapes from the data.
    4. Builds the seq2seq LSTM model using `build_seq2seq_model`.
    5. Trains the model using `train_model` with specified epochs and callbacks.
    6. Saves the final trained model and the best performing model checkpoint.
    7. Prints a summary of the training results.
    """
    # --- Configuration Parameters ---
    # Directory containing the raw NFL data (CSV files).
    train_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    # Number of samples per batch during training and validation.
    batch_size = 32
    # Total number of epochs to train for. The `EarlyStopping` callback may stop training earlier.
    epochs = 5
    # Proportion of the dataset to use for validation.
    test_size = 0.2
    
    print("\n" + "="*60)
    print("NFL Big Data Bowl 2026 - Predictor Training")
    print("="*60)
    
    # --- Data Loading and Preparation ---
    print("\n[1/4] Loading data from CSV files...")
    # Initialize the data loader with the training directory.
    loader = NFLDataLoader(train_dir)
    # Load aligned features (X) and labels (y) from the raw data.
    X, y = loader.get_aligned_data()
    
    # Check if data was loaded successfully.
    if len(X) == 0:
        print("Error: No data loaded. Please check the data directory and ensure CSV files exist.")
        return # Exit the script if no data is found.
    
    # Print summary statistics about the loaded data.
    print(f"\nData Summary:")
    print(f"  Total sequences: {len(X)}")
    # Safely determine sequence lengths and feature counts, handling potential empty lists or sequences.
    if X:
        print(f"  Sample input sequence length: {len(X[0]) if X[0] else 0}")
        print(f"  Input features per timestep: {len(X[0][0]) if len(X[0]) > 0 and X[0][0] else 0}")
    if y:
        print(f"  Sample output sequence length: {len(y[0]) if y[0] else 0}")
        print(f"  Output features per timestep: {len(y[0][0]) if len(y[0]) > 0 and y[0][0] else 0}")
    
    # Create TensorFlow Dataset objects for training and validation.
    # This function handles data splitting, padding, and batching.
    print(f"\n[2/4] Creating training and validation sequences (test_size={test_size}, batch_size={batch_size})...")
    train_seq, val_seq = create_tf_datasets(X, y, test_size=test_size, batch_size=batch_size)
    
    # Check if dataset creation was successful.
    if train_seq is None:
        print("Error: Failed to create training sequences. Check data integrity and parameters.")
        return
    
    # --- Determine Sequence Shapes ---
    # Get a sample batch from the training sequence to infer the exact shapes of input and output.
    # These shapes are required for building the model.
    try:
        x_sample, y_sample = train_seq.take(1).as_numpy_iterator().__next__() # Get one batch
        input_seq_length = x_sample.shape[1] # Number of timesteps in input sequence
        input_features = x_sample.shape[2]   # Number of features per timestep in input
        output_seq_length = y_sample.shape[1] # Number of timesteps in output sequence
        output_features = y_sample.shape[2]  # Number of features per timestep in output
        
        print(f"\nSequence Shapes:")
        print(f"  Input: (batch_size, {input_seq_length}, {input_features})")
        print(f"  Output: (batch_size, {output_seq_length}, {output_features})")
    except Exception as e:
        print(f"Error determining sequence shapes: {e}. Ensure train_seq is not empty.")
        return

    # --- Build Model ---
    print(f"\n[3/4] Building sequence-to-sequence model...")
    # Instantiate the seq2seq model with the determined shapes and default LSTM units.
    model = build_seq2seq_model(
        input_seq_length=input_seq_length,
        input_features=input_features,
        output_seq_length=output_seq_length,
        output_features=output_features,
        lstm_units=128 # This parameter is currently fixed within build_seq2seq_model
    )
    
    print("\nModel Architecture:")
    # Print a summary of the model's layers, parameters, and output shapes.
    model.summary()
    
    # --- Train Model ---
    print(f"\n[4/4] Training model for {epochs} epochs...")
    # Train the model using the prepared sequences and defined number of epochs.
    history = train_model(model, train_seq, val_seq, epochs=epochs)
    
    # --- Save Model ---
    # Save the final trained model (including architecture, weights, and optimizer state).
    final_model_path = 'nfl_predictor_final.keras'
    model.save(final_model_path)
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final model saved to: {final_model_path}")
    # The best model (based on validation loss) is saved by the ModelCheckpoint callback.
    print(f"Best model saved to: best_model.keras")
    print("="*60)
    
    # --- Print Training Summary ---
    # Display key metrics from the training history.
    print(f"\nTraining Summary:")
    if history and history.history:
        print(f"  Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"  Final training MAE: {history.history['mae'][-1]:.4f}")
        print(f"  Final validation MAE: {history.history['val_mae'][-1]:.4f}")
        print(f"  Best validation loss: {min(history.history['val_loss']):.4f}")
    else:
        print("  No training history available to display summary.")

if __name__ == '__main__':
    main()