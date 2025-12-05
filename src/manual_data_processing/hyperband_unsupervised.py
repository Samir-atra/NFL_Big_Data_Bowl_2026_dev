"""
Hyperparameter tuning script for an Unsupervised Sequence-to-Sequence (Seq2Seq) Model.

This module utilizes Keras-Tuner's Hyperband algorithm to find the optimal
hyperparameters for a residual LSTM-based Seq2Seq model. The model is trained
in a self-supervised fashion (e.g., next-step prediction) on combined input data
from both the prediction and analytics datasets.

The final output is the best-performing Seq2Seq model and its encoder part,
which can be used for pre-training feature extraction (unsupervised learning).
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
import keras_tuner as kt

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unsupervised_data_loader import UnsupervisedNFLDataLoader, UnsupervisedNFLSequence

def build_seq2seq_model(hp, 
                        input_seq_length, input_features,
                        output_seq_length, output_features):
    """
    Builds and compiles a residual LSTM Sequence-to-Sequence (Seq2Seq) model
    with tunable hyperparameters for Keras-Tuner.

    The model consists of an encoder that reads the input sequence and produces
    a fixed-size latent vector, and a decoder that translates the latent vector
    into the target output sequence.

    Args:
        hp (kt.HyperParameters): HyperParameters object supplied by Keras-Tuner.
        input_seq_length (int): Length of the input sequences (time steps).
        input_features (int): Number of features in the input sequence.
        output_seq_length (int): Length of the output sequences (time steps).
        output_features (int): Number of features (targets) in the output sequence.

    Returns:
        keras.Model: A compiled Keras model.
    """
    # ---------- Hyper‑parameters ----------
    # Number of LSTM layers (encoder + decoder)
    n_encoder_layers = hp.Int('enc_layers', 2, 4, step=1)
    n_decoder_layers = hp.Int('dec_layers', 2, 4, step=1)

    # LSTM units per layer (same for all layers for simplicity)
    lstm_units = hp.Choice('lstm_units', [64, 128, 256, 384, 512])

    # Dropout rate
    dropout_rate = hp.Float('dropout', 0.0, 0.3, step=0.05)

    # Initial learning rate for the CosineDecay scheduler
    init_lr = hp.Float('init_lr', 1e-4, 5e-3, sampling='log')
    
    # ---------- Model ----------
    # Encoder
    encoder_inputs = layers.Input(shape=(input_seq_length, input_features),
                                  name='encoder_inputs')
    x = encoder_inputs
    # Build residual LSTM encoder stack
    for i in range(n_encoder_layers):
        # Apply LSTM layer
        lstm_out = layers.LSTM(lstm_units,
                               return_sequences=True,
                               name=f'enc_lstm_{i+1}')(x)
        lstm_out = layers.Dropout(dropout_rate,
                                  name=f'enc_dropout_{i+1}')(lstm_out)
        # Add residual connection (if dimensions match)
        if lstm_out.shape[-1] == x.shape[-1]:
            lstm_out = layers.Add(name=f'enc_res_{i+1}')([x, lstm_out])
        # Normalise layer output
        lstm_out = layers.LayerNormalization(name=f'enc_norm_{i+1}')(lstm_out)
        x = lstm_out

    # Grab the final hidden state as the latent vector (context vector)
    latent = layers.LSTM(lstm_units,
                         return_sequences=False,
                         name='latent')(x)

    # Decoder – repeat latent vector for each output timestep
    decoder_inputs = layers.RepeatVector(output_seq_length,
                                         name='repeat_latent')(latent)
    y = decoder_inputs
    # Build residual LSTM decoder stack
    for i in range(n_decoder_layers):
        lstm_out = layers.LSTM(lstm_units,
                               return_sequences=True,
                               name=f'dec_lstm_{i+1}')(y)
        lstm_out = layers.Dropout(dropout_rate,
                                  name=f'dec_dropout_{i+1}')(lstm_out)
        # Residual connection (again only when shapes match)
        if lstm_out.shape[-1] == y.shape[-1]:
            lstm_out = layers.Add(name=f'dec_res_{i+1}')([y, lstm_out])
        lstm_out = layers.LayerNormalization(name=f'dec_norm_{i+1}')(lstm_out)
        y = lstm_out

    # Final TimeDistributed dense layer to map LSTM output to required output features
    decoder_outputs = layers.TimeDistributed(
        layers.Dense(output_features, activation='linear'),
        name='decoder_output')(y)

    model = models.Model(inputs=encoder_inputs, outputs=decoder_outputs,
                         name='tunable_seq2seq')

    # ---------- Learning‑rate schedule ----------
    # Simplified to just CosineDecay
    total_steps = hp.Int('total_steps', 1000, 5000, step=500)
    learning_rate = optimizers.schedules.CosineDecay(
        initial_learning_rate=init_lr,
        decay_steps=total_steps,
        alpha=1e-5)

    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae'])
    return model

def tuner_search(train_seq, val_seq,
                 input_seq_len, input_feat,
                 output_seq_len, output_feat,
                 max_trials=30, epochs_per_trial=5):
    """
    Launches Keras-Tuner's Hyperband search to find the best hyperparameters
    for the unsupervised Seq2Seq model.

    Args:
        train_seq (UnsupervisedNFLSequence): Training data sequence generator.
        val_seq (UnsupervisedNFLSequence): Validation data sequence generator.
        input_seq_len (int): Input sequence length.
        input_feat (int): Input feature count.
        output_seq_len (int): Output sequence length.
        output_feat (int): Output feature count.
        max_trials (int, optional): The total number of trials to run. Defaults to 30.
        epochs_per_trial (int, optional): The initial number of epochs to train a model in Hyperband. Defaults to 5.

    Returns:
        tuple: (best_model, final_history, best_hp)
    """
    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_seq2seq_model(
            hp,
            input_seq_length=input_seq_len,
            input_features=input_feat,
            output_seq_length=output_seq_len,
            output_features=output_feat),
        objective='val_loss',
        max_epochs=epochs_per_trial,
        factor=3,
        directory='kt_tuner_unsupervised',
        project_name='nfl_seq2seq_unsupervised',
        overwrite=True)

    # Early‑stopping inside each trial
    stop_early = callbacks.EarlyStopping(monitor='val_loss',
                                         patience=3,
                                         restore_best_weights=True)

    tuner.search(train_seq,
                 validation_data=val_seq,
                 callbacks=[stop_early],
                 verbose=1)

    # Retrieve the best hyper‑parameters & model
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    # Train the best model a little longer (optional)
    final_history = best_model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs_per_trial * 2,   # give it more epochs now that we know the arch.
        callbacks=[callbacks.EarlyStopping(monitor='val_loss',
                                           patience=5,
                                           restore_best_weights=True)],
        verbose=1)

    return best_model, final_history, best_hp

def main():
    """
    Main function to execute the unsupervised hyperparameter tuning workflow.

    The workflow involves three main steps:
    1. Loading and preparing the combined prediction and analytics data into
       UnsupervisedNFLSequence generators for self-supervised training.
    2. Launching the Keras-Tuner Hyperband search using the defined Seq2Seq model.
    3. Saving the best-performing Seq2Seq model and its encoder component.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Load Unsupervised Data & Prepare Sequences
    # ------------------------------------------------------------------
    PREDICTION_TRAIN_DIR = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    ANALYTICS_TRAIN_DIR = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/'

    print("Loading unsupervised data...")
    # Initialize loader
    loader = UnsupervisedNFLDataLoader()
    # Load from both directories, including both labeled and unlabeled data for unsupervised pre-training
    loader.load_files(
        [PREDICTION_TRAIN_DIR, ANALYTICS_TRAIN_DIR],
        include_labeled=True,
        include_unlabeled=True
    )
    X_unsupervised = loader.get_sequences()

    if len(X_unsupervised) == 0:
        print("ERROR: No unsupervised data loaded!")
        return
    
    print(f"Total unsupervised sequences: {len(X_unsupervised)}")
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val = train_test_split(X_unsupervised, test_size=0.2, random_state=42)
    
    # Create sequences for Next-Step Prediction (Self-Supervised)
    task = 'next_step'
    prediction_steps = 5  # Predict next 5 steps
    
    # Create the training sequence generator
    train_seq = UnsupervisedNFLSequence(
        X_train,
        batch_size=32,
        maxlen=10, 
        shuffle=True,
        task=task,
        prediction_steps=prediction_steps
    )
    
    # Create the validation sequence generator
    val_seq = UnsupervisedNFLSequence(
        X_val,
        batch_size=32,
        maxlen=10,
        shuffle=False,
        task=task,
        prediction_steps=prediction_steps
    )

    # Get shapes from a batch to configure the model
    x_batch, y_batch = train_seq[0]
    input_seq_len = x_batch.shape[1]
    input_feat = x_batch.shape[2]
    output_seq_len = y_batch.shape[1]
    output_feat = y_batch.shape[2]

    print(f"Input shape: ({input_seq_len}, {input_feat})")
    print(f"Output shape: ({output_seq_len}, {output_feat})")

    # ------------------------------------------------------------------
    # 2️⃣  Launch the tuner
    # ------------------------------------------------------------------
    best_model, best_history, best_hp = tuner_search(
            train_seq=train_seq,
            val_seq=val_seq,
            input_seq_len=input_seq_len,
            input_feat=input_feat,
            output_seq_len=output_seq_len,
            output_feat=output_feat,
            max_trials=30,          # increase if you have more time
            epochs_per_trial=12)    # short trials for speed

    print("\n=== Best hyper‑parameters ===")
    for name, value in best_hp.values.items():
        print(f"{name}: {value}")
        
    # ------------------------------------------------------------------
    # 3️⃣  Save final model and encoder
    # ------------------------------------------------------------------
    # Save best model
    best_model.save('best_hyperband_unsupervised_model.keras')
    print("Best model saved to best_hyperband_unsupervised_model.keras")

    # Save encoder separately for pre-training/feature extraction use
    save_encoder_from_model(best_model, 'best_hyperband_encoder.keras')

def save_encoder_from_model(model, path):
    """
    Extracts the encoder part of the Seq2Seq model (up to the 'latent' layer)
    and saves it as a separate Keras model file.

    This encoder model can be used independently for pre-training feature
    extraction or as the encoder for a supervised fine-tuning task.

    Args:
        model (keras.Model): The full Seq2Seq model returned by Keras-Tuner.
        path (str): The file path to save the encoder model to (e.g., 'encoder.keras').
    """
    try:
        # The encoder input is the model input
        encoder_inputs = model.input
        
        # The latent vector is the output of the layer named 'latent'
        latent_layer = model.get_layer('latent')
        latent_output = latent_layer.output
        
        # Create encoder model
        encoder_model = models.Model(inputs=encoder_inputs, outputs=latent_output, name='encoder')
        
        # Save
        encoder_model.save(path)
        print(f"Encoder model saved to {path}")
        return encoder_model
    except Exception as e:
        print(f"Error saving encoder: {e}")
        return None

if __name__ == "__main__":
    main()
