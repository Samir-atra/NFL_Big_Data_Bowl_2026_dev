import tensorflow as tf
from tensorflow import keras
import os
import sys

# Add the manual_data_processing directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'manual_data_processing'))

from csv_to_numpy import NFLDataLoader, create_tf_datasets

def main():
    """
    Load model and evaluate it on the validation data.
    """
    # Configuration
    train_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    batch_size = 32
    test_size = 0.2
    
    print("="*60)
    print("NFL Big Data Bowl 2026 - Model Evaluation")
    print("="*60)
    
    # Load and prepare data
    print("\n[1/3] Loading data from CSV files...")
    loader = NFLDataLoader(train_dir)
    X, y = loader.get_aligned_data()
    
    if len(X) == 0:
        print("Error: No data loaded. Please check the data directory.")
        return
    
    # Create Keras Sequences with padding
    print(f"\n[2/3] Creating training and validation sequences (test_size={test_size})...")
    train_seq, val_seq = create_tf_datasets(X, y, test_size=test_size, batch_size=batch_size)
    
    if train_seq is None:
        print("Error: Failed to create training sequences.")
        return
    
    # Load the best model
    print("\n[3/3] Loading best model...")
    model_path = 'best_model.keras'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
    
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")
    
    # Evaluate on training data
    print("\n" + "="*60)
    print("Evaluating on Training Data")
    print("="*60)
    train_results = model.evaluate(train_seq, verbose=1)
    print(f"\nTraining Loss: {train_results[0]:.6f}")
    print(f"Training MAE: {train_results[1]:.6f}")
    
    # Evaluate on validation data
    print("\n" + "="*60)
    print("Evaluating on Validation Data")
    print("="*60)
    val_results = model.evaluate(val_seq, verbose=1)
    print(f"\nValidation Loss: {val_results[0]:.6f}")
    print(f"Validation MAE: {val_results[1]:.6f}")
    
    # Get a sample prediction to understand the scale
    print("\n" + "="*60)
    print("Sample Prediction Analysis")
    print("="*60)
    x_sample, y_sample = val_seq[0]
    predictions = model.predict(x_sample[:5], verbose=0)
    
    print(f"\nSample ground truth values (y):")
    print(f"Shape: {y_sample[:5].shape}")
    print(f"Mean: {y_sample[:5].mean():.6f}")
    print(f"Std: {y_sample[:5].std():.6f}")
    print(f"Min: {y_sample[:5].min():.6f}")
    print(f"Max: {y_sample[:5].max():.6f}")
    
    print(f"\nSample predictions:")
    print(f"Shape: {predictions.shape}")
    print(f"Mean: {predictions.mean():.6f}")
    print(f"Std: {predictions.std():.6f}")
    print(f"Min: {predictions.min():.6f}")
    print(f"Max: {predictions.max():.6f}")
    
    # Calculate sample MSE
    sample_mse = ((predictions - y_sample[:5]) ** 2).mean()
    print(f"\nSample MSE: {sample_mse:.6f}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)

if __name__ == '__main__':
    main()
