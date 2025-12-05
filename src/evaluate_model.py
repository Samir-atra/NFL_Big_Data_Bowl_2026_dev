"""
This module is responsible for evaluating the trained machine learning model for the
NFL Big Data Bowl 2026 competition. It performs the following key tasks:

1.  **Data Loading:** Loads preprocessed data (features X and labels y) from CSV files.
2.  **Sequence Creation:** Creates TensorFlow `Sequence` objects for training and validation, handling data padding and batching.
3.  **Model Loading:** Loads the pre-trained Keras model from a specified file.
4.  **Evaluation:** Evaluates the loaded model on both the training and validation datasets to assess its performance (loss and Mean Absolute Error).
5.  **Sample Prediction Analysis:** Generates sample predictions on a subset of validation data to understand the model's output scale and characteristics.

This script assumes that the necessary data files and the trained model file (`best_model.keras`) are available in the specified paths.
"""

import tensorflow as tf
from tensorflow import keras
import os
import sys

# Add the manual_data_processing directory to the Python path to import custom modules.
# This is necessary because 'csv_to_numpy.py' is located in a subdirectory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'manual_data_processing'))

# Import necessary components from the custom module.
from csv_to_numpy import NFLDataLoader, create_tf_datasets

def main():
    """
    Main function to load a pre-trained Keras model, prepare the dataset,
    and evaluate the model's performance on training and validation sets.

    The evaluation includes calculating loss and Mean Absolute Error (MAE),
    and performing a sample prediction analysis to understand the output scale.
    """
    # --- Configuration --- 
    # Directory containing the training data (CSV files).
    train_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    # Batch size for creating TensorFlow datasets.
    batch_size = 32
    # Proportion of data to be used for validation.
    test_size = 0.2
    
    print("\n" + "="*60)
    print("NFL Big Data Bowl 2026 - Model Evaluation")
    print("="*60)
    
    # --- Data Loading and Preparation --- 
    print("\n[1/3] Loading data from CSV files...")
    # Initialize the NFLDataLoader to load data from the specified directory.
    loader = NFLDataLoader(train_dir)
    # Get the aligned features (X) and labels (y) from the loaded data.
    X, y = loader.get_aligned_data()
    
    # Check if any data was loaded successfully.
    if len(X) == 0:
        print("Error: No data loaded. Please check the data directory and ensure CSV files exist.")
        return # Exit if no data is available
    
    # Create TensorFlow datasets for training and validation.
    print(f"\n[2/3] Creating training and validation sequences (test_size={test_size}, batch_size={batch_size})...")
    # `create_tf_datasets` handles splitting, padding, and batching.
    train_seq, val_seq = create_tf_datasets(X, y, test_size=test_size, batch_size=batch_size)
    
    # Check if dataset creation was successful.
    if train_seq is None or val_seq is None:
        print("Error: Failed to create training or validation sequences. Check data integrity and parameters.")
        return
    
    # --- Model Loading --- 
    print("\n[3/3] Loading best model...")
    # Define the path to the pre-trained model file.
    model_path = 'best_model.keras'
    
    # Check if the model file exists before attempting to load.
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please ensure the model file is in the correct directory.")
        return
    
    # Load the Keras model from the specified path.
    try:
        model = keras.models.load_model(model_path)
        print(f"Model '{model_path}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        return

    # --- Evaluation on Training Data --- 
    print("\n" + "="*60)
    print("Evaluating on Training Data")
    print("="*60)
    
    # Evaluate the model using the training dataset.
    # `model.evaluate` returns a list of metrics (e.g., loss, MAE).
    try:
        train_results = model.evaluate(train_seq, verbose=1)
        # Assuming the first metric is loss and the second is MAE (common for regression tasks).
        print(f"\nTraining Loss: {train_results[0]:.6f}")
        print(f"Training MAE: {train_results[1]:.6f}")
    except Exception as e:
        print(f"Error during training data evaluation: {e}")

    # --- Evaluation on Validation Data --- 
    print("\n" + "="*60)
    print("Evaluating on Validation Data")
    print("="*60)
    
    # Evaluate the model using the validation dataset.
    try:
        val_results = model.evaluate(val_seq, verbose=1)
        print(f"\nValidation Loss: {val_results[0]:.6f}")
        print(f"Validation MAE: {val_results[1]:.6f}")
    except Exception as e:
        print(f"Error during validation data evaluation: {e}")

    # --- Sample Prediction Analysis --- 
    print("\n" + "="*60)
    print("Sample Prediction Analysis")
    print("="*60)
    
    # Get a small batch of data from the validation sequence for sample predictions.
    # Using verbose=0 to avoid printing progress bars during prediction.
    try:
        # Fetch a sample batch
        x_sample, y_sample = val_seq[0] # Takes the first batch
        
        # Get predictions for the first 5 samples in the batch.
        predictions = model.predict(x_sample[:5], verbose=0)
        
        # Display statistics for the ground truth values of the sample batch.
        print(f"\nSample ground truth values (y):")
        print(f"Shape: {y_sample[:5].shape}")
        print(f"Mean: {y_sample[:5].mean():.6f}")
        print(f"Std: {y_sample[:5].std():.6f}")
        print(f"Min: {y_sample[:5].min():.6f}")
        print(f"Max: {y_sample[:5].max():.6f}")
        
        # Display statistics for the model's predictions on the sample batch.
        print(f"\nSample predictions:")
        print(f"Shape: {predictions.shape}")
        print(f"Mean: {predictions.mean():.6f}")
        print(f"Std: {predictions.std():.6f}")
        print(f"Min: {predictions.min():.6f}")
        print(f"Max: {predictions.max():.6f}")
        
        # Calculate and print the Mean Squared Error (MSE) for the sample predictions.
        sample_mse = tf.keras.losses.mean_squared_error(y_sample[:5], predictions).numpy().mean()
        print(f"\nSample MSE: {sample_mse:.6f}")
        
    except Exception as e:
        print(f"Error during sample prediction analysis: {e}")

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)

if __name__ == '__main__':
    main()
