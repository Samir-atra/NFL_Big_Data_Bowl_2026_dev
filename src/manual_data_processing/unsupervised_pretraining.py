"""
Unsupervised Pre-training Script for NFL Player Trajectory Prediction

This script performs unsupervised pre-training using LSTM autoencoders on all available
NFL player sequences (both labeled and unlabeled). The pretrained encoder can then be
used to initialize supervised models for better performance.

Usage:
    python unsupervised_pretraining.py --task autoencoder --epochs 50
    python unsupervised_pretraining.py --task next_step --epochs 50
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unsupervised_data_loader import UnsupervisedNFLDataLoader, UnsupervisedNFLSequence
from unsupervised_models import (
    LSTMAutoencoder, 
    NextStepPredictor, 
    create_training_callbacks
)


def train_autoencoder(train_seq, val_seq, epochs=50, latent_dim=128, model_save_path='autoencoder.keras'):
    """
    Trains an LSTM autoencoder for sequence representation learning (reconstruction task).
    
    The function builds, compiles, and trains the model, and then saves both the 
    full autoencoder and its encoder component.
    
    Args:
        train_seq (UnsupervisedNFLSequence): Training data sequence generator.
        val_seq (UnsupervisedNFLSequence): Validation data sequence generator.
        epochs (int): Number of training epochs.
        latent_dim (int): Dimension of the latent space for the encoder output.
        model_save_path (str): Path to save the best full autoencoder model.
        
    Returns:
        tuple: (ae, history) where ae is the trained LSTMAutoencoder instance.
    """
    print("\n" + "="*70)
    print("TRAINING LSTM AUTOENCODER")
    print("="*70)
    
    # Get input shape from a sample batch to configure the model architecture
    x_sample, _ = train_seq[0]
    input_shape = (x_sample.shape[1], x_sample.shape[2])
    
    print(f"\nInput shape: {input_shape}")
    print(f"Latent dimension: {latent_dim}")
    
    # Build autoencoder model
    ae = LSTMAutoencoder(
        input_shape=input_shape,
        latent_dim=latent_dim,
        lstm_units=[256, 128] # Predefined LSTM unit architecture
    )
    ae.build_autoencoder()
    ae.compile(learning_rate=0.001)
    
    print("\n" + "-"*70)
    ae.get_summary()
    
    # Create callbacks for checkpointing and early stopping
    callbacks = create_training_callbacks(model_save_path, patience=10)
    
    # Train the model
    print("\n" + "-"*70)
    print("Starting training...")
    print("-"*70)
    
    history = ae.autoencoder.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Model saved to: {model_save_path}")
    print("="*70)
    
    # Save encoder separately for use in transfer learning
    encoder_path = model_save_path.replace('.keras', '_encoder.keras')
    ae.encoder.save(encoder_path)
    print(f"Encoder saved to: {encoder_path}")
    
    return ae, history


def train_next_step_predictor(train_seq, val_seq, epochs=50, prediction_steps=5, 
                               model_save_path='next_step_predictor.keras'):
    """
    Trains a NextStepPredictor model for self-supervised sequence prediction.
    
    The model is trained to predict the next 'N' steps given the preceding sequence.
    
    Args:
        train_seq (UnsupervisedNFLSequence): Training data sequence generator.
        val_seq (UnsupervisedNFLSequence): Validation data sequence generator.
        epochs (int): Number of training epochs.
        prediction_steps (int): Number of steps the model is configured to predict ahead.
        model_save_path (str): Path to save the trained model.
        
    Returns:
        tuple: (predictor, history) where predictor is the trained NextStepPredictor instance.
    """
    print("\n" + "="*70)
    print("TRAINING NEXT-STEP PREDICTOR")
    print("="*70)
    
    # Get input and output shapes from a sample batch
    x_sample, y_sample = train_seq[0]
    input_shape = (x_sample.shape[1], x_sample.shape[2])
    output_features = y_sample.shape[2]
    
    print(f"\nInput shape: {input_shape}")
    print(f"Output steps: {prediction_steps}")
    print(f"Output features: {output_features}")
    
    # Build prediction model
    predictor = NextStepPredictor(
        input_shape=input_shape,
        output_steps=prediction_steps,
        lstm_units=[256, 128], # Predefined LSTM unit architecture
        output_features=output_features
    )
    predictor.build()
    predictor.compile(learning_rate=0.001)
    
    print("\n" + "-"*70)
    predictor.get_summary()
    
    # Create callbacks
    callbacks = create_training_callbacks(model_save_path, patience=10)
    
    # Train
    print("\n" + "-"*70)
    print("Starting training...")
    print("-"*70)
    
    history = predictor.model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Model saved to: {model_save_path}")
    print("="*70)
    
    return predictor, history


def main():
    """
    Parses command-line arguments, loads and splits the unsupervised data, 
    creates data generators, and launches the specified unsupervised training task.
    """
    parser = argparse.ArgumentParser(description='Unsupervised Pre-training for NFL Data')
    parser.add_argument('--task', type=str, default='autoencoder', 
                       choices=['autoencoder', 'next_step'],
                       help='Unsupervised task to train (autoencoder or next_step)')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--maxlen', type=int, default=None, 
                       help='Maximum sequence length (auto-detect if None)')
    parser.add_argument('--latent_dim', type=int, default=128, 
                       help='Latent dimension for autoencoder')
    parser.add_argument('--prediction_steps', type=int, default=5, 
                       help='Number of steps to predict for next_step task')
    parser.add_argument('--val_split', type=float, default=0.2, 
                       help='Validation split ratio')
    parser.add_argument('--include_labeled', action='store_true', default=True,
                       help='Include labeled data (player_to_predict=True)')
    parser.add_argument('--include_unlabeled', action='store_true', default=True,
                       help='Include unlabeled data (player_to_predict=False)')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/models/unsupervised',
                       help='Directory to save trained models')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data directories (using both prediction and analytics data for broader coverage)
    PREDICTION_TRAIN_DIR = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    ANALYTICS_TRAIN_DIR = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/'
    
    print("\n" + "="*70)
    print("UNSUPERVISED PRE-TRAINING FOR NFL PLAYER TRAJECTORY PREDICTION")
    print("="*70)
    print(f"\nTask: {args.task}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Include labeled: {args.include_labeled}")
    print(f"Include unlabeled: {args.include_unlabeled}")
    print(f"Validation split: {args.val_split}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    loader = UnsupervisedNFLDataLoader()
    loader.load_files(
        [PREDICTION_TRAIN_DIR, ANALYTICS_TRAIN_DIR],
        include_labeled=args.include_labeled,
        include_unlabeled=args.include_unlabeled
    )
    X = loader.get_sequences()
    
    if len(X) == 0:
        print("ERROR: No data loaded! Exiting.")
        return
    
    print(f"\nTotal sequences loaded: {len(X)}")
    # Note: X is a numpy object array, accessing X[0] gives the first sequence (T, F)
    print(f"Sample sequence length: {len(X[0])}")
    print(f"Sample features: {len(X[0][0])}")
    
    # Split loaded sequences into training and validation sets
    from sklearn.model_selection import train_test_split
    
    X_train, X_val = train_test_split(
        X, 
        test_size=args.val_split, 
        random_state=42
    )
    
    print(f"\nTraining sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")
    
    # Create data sequences (generators) based on the chosen task
    print("\n" + "="*70)
    print("CREATING DATA GENERATORS")
    print("="*70)
    
    train_seq = UnsupervisedNFLSequence(
        X_train,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        shuffle=True,
        task=args.task,
        prediction_steps=args.prediction_steps
    )
    
    val_seq = UnsupervisedNFLSequence(
        X_val,
        batch_size=args.batch_size,
        # Use the maximum length calculated by the training sequence generator
        maxlen=train_seq.maxlen, 
        shuffle=False,
        task=args.task,
        prediction_steps=args.prediction_steps
    )
    
    # Generate timestamp for model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Launch training based on task
    if args.task == 'autoencoder':
        model_path = os.path.join(args.output_dir, f'autoencoder_{timestamp}.keras')
        model, history = train_autoencoder(
            train_seq, 
            val_seq, 
            epochs=args.epochs,
            latent_dim=args.latent_dim,
            model_save_path=model_path
        )
        
    elif args.task == 'next_step':
        model_path = os.path.join(args.output_dir, f'next_step_{timestamp}.keras')
        model, history = train_next_step_predictor(
            train_seq,
            val_seq,
            epochs=args.epochs,
            prediction_steps=args.prediction_steps,
            model_save_path=model_path
        )
    
    # Print final summary and instructions for transfer learning
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"\nModel saved to: {model_path}")
    
    if args.task == 'autoencoder':
        encoder_path = model_path.replace('.keras', '_encoder.keras')
        print(f"Encoder saved to: {encoder_path}")
        print("\nTo use the pretrained encoder in your supervised model:")
        print(f"  from tensorflow import keras")
        print(f"  from unsupervised_models import transfer_encoder_weights")
        print(f"  pretrained_encoder = keras.models.load_model('{encoder_path}')")
        print(f"  supervised_model = transfer_encoder_weights(pretrained_encoder, supervised_model)")
    
    print("="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()