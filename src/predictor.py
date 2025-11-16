import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from data_loader import load_and_prepare_data, SEQUENCE_LENGTH

import joblib
def build_model(input_features, output_shape, lstm_units=64):
    """
    Builds a sequential model with two LSTM layers.

    Args:
        input_features (int): The number of input features per timestep.
        output_shape (int): The number of output units.
        lstm_units (int): The number of units in the LSTM layers.

    Returns:
        keras.Model: The compiled Keras model.
    """
    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, input_features)),  # Input shape for a sequence of timesteps
        layers.LSTM(lstm_units),
        layers.Dense(output_shape)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model

def train_model(model, train_dataset, val_dataset, epochs, batch_size):
    """
    Trains the Keras model.
    """
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    if val_dataset:
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("Starting model training...")
    history = model.fit(train_dataset,
                        epochs=epochs,
                        validation_data=val_dataset)
    print("Model training finished.")
    return history

def main():
    """
    Main function to load data, build, and train the model.
    """
    prediction_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train'
    
    batch_size = 32
    epochs = 10

    train_ds, val_ds, preprocessor = load_and_prepare_data(prediction_data_dir)

    if train_ds.cardinality().numpy() == 0:
        print("No training data generated. Please check data loading and feature engineering.")
        return

    # Get the input and output shapes from the dataset specs
    feature_spec, label_spec = train_ds.element_spec
    input_features = feature_spec.shape[1] # Now shape is (SEQUENCE_LENGTH, input_features)
    output_shape = label_spec.shape[0]

    model = build_model(input_features, output_shape)
    model.summary()

    train_model(model, train_ds, val_ds, epochs, batch_size)

    # Save the trained model and the preprocessor
    model_save_path = 'nfl_model.h5'
    preprocessor_save_path = 'preprocessor.joblib'

    model.save(model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)

    print(f"Model saved to {model_save_path}")
    print(f"Preprocessor saved to {preprocessor_save_path}")


if __name__ == '__main__':
    main()