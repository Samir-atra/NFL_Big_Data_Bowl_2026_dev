import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys

# Add the manual_data_processing directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'manual_data_processing'))

from csv_to_numpy import NFLDataLoader, create_tf_datasets

def build_seq2seq_model(input_seq_length, input_features, output_seq_length, output_features, lstm_units=128):
    """
    Builds a sequence-to-sequence model with LSTM layers.

    Args:
        input_seq_length (int): The length of input sequences (time steps).
        input_features (int): The number of input features per timestep.
        output_seq_length (int): The length of output sequences (time steps).
        output_features (int): The number of output features per timestep.
        lstm_units (int): The number of units in the LSTM layers.

    Returns:
        keras.Model: The compiled Keras model.
    """

    SEED = 42
    # Encoder-decoder architecture for sequence-to-sequence prediction
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_seq_length, input_features)),
        
        keras.layers.LSTM(
            units=123,
            activation="sigmoid", # Fixed activation for the first layer
            return_sequences=True,
            # kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            seed=SEED,
        ),
        keras.layers.LSTM(
            units=64,
            activation="sigmoid", # Tunable activation function
            return_sequences=True,
            # kernel_regularizer=keras.regularizers.L2(l2=kernel_r), # Tunable kernel regularization
            seed=SEED,
        ),
        keras.layers.LSTM(
            units=64, # Tunable number of units
            activation="sigmoid",
            return_sequences=True,
            # kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
            seed=SEED,
        ),
        keras.layers.LSTM(
            units=32,
            activation="sigmoid",
            return_sequences=True,
            # kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            seed=SEED,
        ),
        # Crop or slice to match output sequence length
        layers.Lambda(lambda x: x[:, :output_seq_length, :]),
        # TimeDistributed dense layer for output features
        layers.TimeDistributed(keras.layers.Dense(units=output_features, activation="linear")),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(model, train_sequence, val_sequence, epochs=10, callbacks=None):
    """
    Trains the Keras model using Keras Sequence objects.
    
    Args:
        model: The Keras model to train
        train_sequence: Training data sequence (NFLDataSequence)
        val_sequence: Validation data sequence (NFLDataSequence)
        epochs (int): Number of training epochs
        callbacks: List of Keras callbacks
    
    Returns:
        history: Training history object
    """
    if callbacks is None:
        callbacks = []
    
    # Add early stopping and model checkpoint callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    callbacks.extend([early_stopping, model_checkpoint])
    
    print("Starting model training...")
    history = model.fit(
        train_sequence,
        epochs=epochs,
        validation_data=val_sequence,
        callbacks=callbacks,
        verbose=1
    )
    print("Model training finished.")
    return history

def main():
    """
    Main function to load data, build, and train the model.
    """
    # Configuration
    train_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    batch_size = 32
    epochs = 5
    test_size = 0.2
    
    print("="*60)
    print("NFL Big Data Bowl 2026 - Predictor Training")
    print("="*60)
    
    # Load and prepare data
    print("\n[1/4] Loading data from CSV files...")
    loader = NFLDataLoader(train_dir)
    X, y = loader.get_aligned_data()
    
    if len(X) == 0:
        print("Error: No data loaded. Please check the data directory.")
        return
    
    print(f"\nData Summary:")
    print(f"  Total sequences: {len(X)}")
    print(f"  Sample input sequence length: {len(X[0])}")
    print(f"  Sample output sequence length: {len(y[0])}")
    print(f"  Input features per timestep: {len(X[0][0]) if len(X[0]) > 0 else 0}")
    print(f"  Output features per timestep: {len(y[0][0]) if len(y[0]) > 0 else 0}")
    
    # Create Keras Sequences with padding
    print(f"\n[2/4] Creating training and validation sequences (test_size={test_size})...")
    train_seq, val_seq = create_tf_datasets(X, y, test_size=test_size, batch_size=batch_size)
    
    if train_seq is None:
        print("Error: Failed to create training sequences.")
        return
    
    # Get one batch to determine shapes
    x_sample, y_sample = train_seq[0]
    input_seq_length = x_sample.shape[1]
    input_features = x_sample.shape[2]
    output_seq_length = y_sample.shape[1]
    output_features = y_sample.shape[2]
    
    print(f"\nSequence Shapes:")
    print(f"  Input: (batch_size, {input_seq_length}, {input_features})")
    print(f"  Output: (batch_size, {output_seq_length}, {output_features})")
    
    # Build model
    print(f"\n[3/4] Building sequence-to-sequence model...")
    model = build_seq2seq_model(
        input_seq_length=input_seq_length,
        input_features=input_features,
        output_seq_length=output_seq_length,
        output_features=output_features,
        lstm_units=128
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print(f"\n[4/4] Training model for {epochs} epochs...")
    history = train_model(model, train_seq, val_seq, epochs=epochs)
    
    # Save the final model
    final_model_path = 'nfl_predictor_final.keras'
    model.save(final_model_path)
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: best_model.keras")
    print(f"{'='*60}")
    
    # Print training summary
    print(f"\nTraining Summary:")
    print(f"  Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  Final training MAE: {history.history['mae'][-1]:.4f}")
    print(f"  Final validation MAE: {history.history['val_mae'][-1]:.4f}")
    print(f"  Best validation loss: {min(history.history['val_loss']):.4f}")

if __name__ == '__main__':
    main()