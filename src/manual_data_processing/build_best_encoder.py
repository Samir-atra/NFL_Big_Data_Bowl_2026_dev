"""
Build the best encoder model found by Hyperband tuning.

This script recreates the exact architecture from the best hyperparameter search:
- 6 encoder LSTM layers with 512 units each
- Dropout rate: 0.2
- Residual connections and Layer Normalization
- Input: (10, 18) - 10 timesteps, 18 features
- Output: (512,) - latent representation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def build_best_encoder(input_seq_length=10, input_features=18, lstm_units=512, dropout_rate=0.2):
    """
    Best Encoder Architecture from Hyperband Tuning
    
    Hyperparameters:
    - enc_layers: 6
    - lstm_units: 512
    - dropout: 0.2
    - init_lr: 0.0007976042571981798
    - total_steps: 86898
    """

    encoder_inputs = layers.Input(shape=(input_seq_length, input_features), name='encoder_inputs')
    
    enc_lstm_1 = layers.LSTM(lstm_units, return_sequences=True, name='enc_lstm_1')(encoder_inputs)
    enc_dropout_1 = layers.Dropout(dropout_rate, name='enc_dropout_1')(enc_lstm_1)
    enc_norm_1 = layers.LayerNormalization(name='enc_norm_1')(enc_dropout_1)
    
    enc_lstm_2 = layers.LSTM(lstm_units, return_sequences=True, name='enc_lstm_2')(enc_norm_1)
    enc_dropout_2 = layers.Dropout(dropout_rate, name='enc_dropout_2')(enc_lstm_2)
    enc_res_2 = layers.Add(name='enc_res_2')([enc_norm_1, enc_dropout_2])
    enc_norm_2 = layers.LayerNormalization(name='enc_norm_2')(enc_res_2)
    
    enc_lstm_3 = layers.LSTM(lstm_units, return_sequences=True, name='enc_lstm_3')(enc_norm_2)
    enc_dropout_3 = layers.Dropout(dropout_rate, name='enc_dropout_3')(enc_lstm_3)
    enc_res_3 = layers.Add(name='enc_res_3')([enc_norm_2, enc_dropout_3])
    enc_norm_3 = layers.LayerNormalization(name='enc_norm_3')(enc_res_3)
    
    enc_lstm_4 = layers.LSTM(lstm_units, return_sequences=True, name='enc_lstm_4')(enc_norm_3)
    enc_dropout_4 = layers.Dropout(dropout_rate, name='enc_dropout_4')(enc_lstm_4)
    enc_res_4 = layers.Add(name='enc_res_4')([enc_norm_3, enc_dropout_4])
    enc_norm_4 = layers.LayerNormalization(name='enc_norm_4')(enc_res_4)
    
    enc_lstm_5 = layers.LSTM(lstm_units, return_sequences=True, name='enc_lstm_5')(enc_norm_4)
    enc_dropout_5 = layers.Dropout(dropout_rate, name='enc_dropout_5')(enc_lstm_5)
    enc_res_5 = layers.Add(name='enc_res_5')([enc_norm_4, enc_dropout_5])
    enc_norm_5 = layers.LayerNormalization(name='enc_norm_5')(enc_res_5)
    
    enc_lstm_6 = layers.LSTM(lstm_units, return_sequences=True, name='enc_lstm_6')(enc_norm_5)
    enc_dropout_6 = layers.Dropout(dropout_rate, name='enc_dropout_6')(enc_lstm_6)
    enc_res_6 = layers.Add(name='enc_res_6')([enc_norm_5, enc_dropout_6])
    enc_norm_6 = layers.LayerNormalization(name='enc_norm_6')(enc_res_6)
    
    latent = layers.LSTM(lstm_units, return_sequences=False, name='latent')(enc_norm_6)
    
    encoder = Model(inputs=encoder_inputs, outputs=latent, name='encoder')
    
    return encoder


if __name__ == "__main__":
    print("=" * 70)
    print("Building Best Encoder Architecture")
    print("=" * 70)
    
    # Build the encoder
    encoder = build_best_encoder()
    
    print("\nEncoder Model Summary:")
    print("-" * 70)
    encoder.summary()
    
    print("\n" + "=" * 70)
    print("Model Architecture Details")
    print("=" * 70)
    print(f"Total parameters: {encoder.count_params():,}")
    print(f"Input shape: (batch_size, 10, 18)")
    print(f"Output shape: (batch_size, 512)")
    print(f"Number of encoder layers: 6")
    print(f"LSTM units per layer: 512")
    print(f"Dropout rate: 0.2")
    
    # Test forward pass
    print("\n" + "=" * 70)
    print("Testing Forward Pass")
    print("=" * 70)
    
    import numpy as np
    test_input = np.random.randn(4, 10, 18).astype(np.float32)
    print(f"Test input shape: {test_input.shape}")
    
    latent_output = encoder.predict(test_input, verbose=0)
    print(f"Latent output shape: {latent_output.shape}")
    
    # Save the model
    output_path = 'encoder_architecture.keras'
    encoder.save(output_path)
    print(f"\nEncoder saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("To load the pretrained weights from Hyperband:")
    print("=" * 70)
    print("from tensorflow import keras")
    print("pretrained_encoder = keras.models.load_model('best_hyperband_encoder.keras')")
    print("# Use pretrained_encoder for inference or transfer learning")
