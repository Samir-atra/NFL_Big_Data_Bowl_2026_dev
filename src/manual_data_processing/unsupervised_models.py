"""
Module defining Keras models for unsupervised pre-training on sequential NFL data.

This module provides two main models:
1. LSTMAutoencoder: For learning a compressed latent representation via sequence reconstruction.
2. NextStepPredictor: For learning temporal dependencies via a sequence-to-sequence prediction task.

It also includes utility functions for configuring standard training callbacks and
transferring weights from a pre-trained encoder to a downstream supervised model.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class LSTMAutoencoder:
    """LSTM Autoencoder for unsupervised representation learning on NFL sequences.
    
    The encoder learns to compress player movement sequences into a latent representation,
    and the decoder reconstructs the original sequence. The encoder can then be used
    to initialize supervised models.
    """
    
    def __init__(self, input_shape, latent_dim=128, lstm_units=[256, 128]):
        """
        Initialize the LSTM Autoencoder.
        
        Args:
            input_shape (tuple): Shape of input sequences (timesteps, features).
            latent_dim (int): Dimension of the compressed latent representation.
            lstm_units (list): List of LSTM units for encoder/decoder layers (e.g., [256, 128] for 2 layers).
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        
    def build_encoder(self):
        """
        Builds the encoder network which maps the input sequence to a latent vector.

        The encoder uses a stack of LSTM layers, where the last layer outputs only
        the final hidden state (return_sequences=False).

        Returns:
            keras.Model: The compiled encoder model.
        """
        inputs = layers.Input(shape=self.input_shape, name='encoder_input')
        
        x = inputs
        # Stack LSTM layers (all but the last return sequences for stacking)
        for i, units in enumerate(self.lstm_units[:-1]):
            x = layers.LSTM(
                units, 
                return_sequences=True,
                name=f'encoder_lstm_{i+1}'
            )(x)
            x = layers.Dropout(0.2)(x)
        
        # Last LSTM layer doesn't return sequences, producing the context vector
        x = layers.LSTM(
            self.lstm_units[-1],
            return_sequences=False,
            name=f'encoder_lstm_{len(self.lstm_units)}'
        )(x)
        x = layers.Dropout(0.2)(x)
        
        # Latent representation layer (bottleneck)
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        self.encoder = Model(inputs, latent, name='encoder')
        return self.encoder
    
    def build_decoder(self):
        """
        Builds the decoder network which maps the latent vector back to a sequence.

        The decoder starts by repeating the latent vector for each output timestep.
        It uses a stack of LSTM layers, followed by a TimeDistributed Dense layer
        for feature reconstruction at every time step.

        Returns:
            keras.Model: The compiled decoder model.
        """
        # Decoder input is the latent vector
        latent_inputs = layers.Input(shape=(self.latent_dim,), name='decoder_input')
        
        # Repeat the latent vector for each timestep (required for LSTM decoding)
        x = layers.RepeatVector(self.input_shape[0])(latent_inputs)
        
        # Stack LSTM layers (return sequences=True for sequence output)
        for i, units in enumerate(reversed(self.lstm_units)):
            x = layers.LSTM(
                units,
                return_sequences=True,
                name=f'decoder_lstm_{i+1}'
            )(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer to reconstruct features at every timestep
        outputs = layers.TimeDistributed(
            layers.Dense(self.input_shape[1], activation='linear'),
            name='reconstruction'
        )(x)
        
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        return self.decoder
    
    def build_autoencoder(self):
        """
        Builds and connects the encoder and decoder to form the complete autoencoder.

        Returns:
            keras.Model: The full autoencoder model.
        """
        if self.encoder is None:
            self.build_encoder()
        if self.decoder is None:
            self.build_decoder()
        
        # Connect encoder and decoder
        inputs = layers.Input(shape=self.input_shape, name='autoencoder_input')
        latent = self.encoder(inputs)
        outputs = self.decoder(latent)
        
        self.autoencoder = Model(inputs, outputs, name='autoencoder')
        return self.autoencoder
    
    def compile(self, learning_rate=0.001):
        """
        Compiles the autoencoder model with the Adam optimizer and Mean Squared Error (MSE) loss.

        Args:
            learning_rate (float, optional): The learning rate for the Adam optimizer. Defaults to 0.001.
        """
        if self.autoencoder is None:
            self.build_autoencoder()
        
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
    def get_summary(self):
        """Prints the model summaries for the autoencoder, encoder, and decoder."""
        if self.autoencoder:
            print("\n=== Autoencoder Summary ===")
            self.autoencoder.summary()
        if self.encoder:
            print("\n=== Encoder Summary ===")
            self.encoder.summary()
        if self.decoder:
            print("\n=== Decoder Summary ===")
            self.decoder.summary()


class NextStepPredictor:
    """LSTM model for self-supervised next-step prediction.
    
    This model is trained to predict future timesteps given past timesteps, 
    serving as a self-supervised pre-training task to learn temporal dependencies 
    in the player movement sequences.
    """
    
    def __init__(self, input_shape, output_steps=5, lstm_units=[256, 128], output_features=None):
        """
        Initialize the next-step predictor.
        
        Args:
            input_shape (tuple): Shape of the input sequences (timesteps, features).
            output_steps (int): Number of future steps to predict (the length of the output sequence).
            lstm_units (list): List of LSTM units for the encoder stack.
            output_features (int, optional): Number of output features. If None, it is set 
                                             to be the same as the input features.
        """
        self.input_shape = input_shape
        self.output_steps = output_steps
        self.lstm_units = lstm_units
        self.output_features = output_features or input_shape[1]
        self.model = None
        
    def build(self):
        """
        Builds the sequence-to-sequence prediction model.

        The architecture uses a stacked LSTM encoder followed by a RepeatVector
        and another LSTM layer to decode the future sequence.

        Returns:
            keras.Model: The uncompiled prediction model.
        """
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        x = inputs
        # Stack LSTM encoder layers
        for i, units in enumerate(self.lstm_units):
            # Only the final LSTM layer returns sequences=False
            return_seq = (i < len(self.lstm_units) - 1)
            x = layers.LSTM(
                units,
                return_sequences=return_seq,
                name=f'lstm_{i+1}'
            )(x)
            x = layers.Dropout(0.2)(x)
        
        # Prediction head: repeat the final state (context vector)
        x = layers.RepeatVector(self.output_steps)(x)
        # Decoding LSTM
        x = layers.LSTM(128, return_sequences=True, name='prediction_lstm')(x)
        
        # Output for each timestep using TimeDistributed Dense layer
        outputs = layers.TimeDistributed(
            layers.Dense(self.output_features, activation='linear'),
            name='predictions'
        )(x)
        
        self.model = Model(inputs, outputs, name='next_step_predictor')
        return self.model
    
    def compile(self, learning_rate=0.001):
        """
        Compiles the model with the Adam optimizer and Mean Squared Error (MSE) loss.

        Args:
            learning_rate (float, optional): The learning rate for the Adam optimizer. Defaults to 0.001.
        """
        if self.model is None:
            self.build()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def get_summary(self):
        """Prints the model summary."""
        if self.model:
            self.model.summary()


def create_training_callbacks(model_path, patience=10):
    """
    Creates a standard list of Keras callbacks for robust training.

    Includes EarlyStopping, ModelCheckpoint to save the best model, and
    ReduceLROnPlateau for dynamic learning rate adjustments.

    Args:
        model_path (str): Path to save the best model weights (for ModelCheckpoint).
        patience (int, optional): Patience value for EarlyStopping. Defaults to 10.
        
    Returns:
        list: A list of configured Keras callback objects.
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks


def transfer_encoder_weights(pretrained_encoder, supervised_model, freeze_encoder=False):
    """
    Transfers weights from a pre-trained encoder model (e.g., from an Autoencoder or 
    NextStepPredictor) to the matching layers in a new supervised model.

    This is a key step in fine-tuning a model after unsupervised pre-training.

    Args:
        pretrained_encoder (keras.Model): The model containing the pre-trained encoder layers.
        supervised_model (keras.Model): The new model that is receiving the weights.
        freeze_encoder (bool, optional): Whether to set the transferred layers to non-trainable. Defaults to False.
        
    Returns:
        keras.Model: The supervised model with transferred weights.
    """
    print("\n=== Transferring Encoder Weights ===")
    
    # Get encoder layers from pretrained model
    encoder_layer_names = [layer.name for layer in pretrained_encoder.layers]
    
    # Transfer weights to matching layers in supervised model
    transferred_count = 0
    for layer in supervised_model.layers:
        if layer.name in encoder_layer_names:
            try:
                pretrained_layer = pretrained_encoder.get_layer(layer.name)
                layer.set_weights(pretrained_layer.get_weights())
                
                if freeze_encoder:
                    layer.trainable = False
                
                transferred_count += 1
                print(f"Transferred weights for layer: {layer.name} (frozen={freeze_encoder})")
            except Exception as e:
                print(f"Could not transfer weights for {layer.name}: {e}")
    
    print(f"\nTransferred weights for {transferred_count} layers")
    return supervised_model


if __name__ == "__main__":
    print("=== Testing Unsupervised Models ===\n")
    
    # Test parameters
    timesteps = 28
    features = 18
    latent_dim = 64
    
    print("1. Testing LSTM Autoencoder")
    print("-" * 50)
    ae = LSTMAutoencoder(
        input_shape=(timesteps, features),
        latent_dim=latent_dim,
        lstm_units=[128, 64]
    )
    ae.build_autoencoder()
    ae.compile()
    ae.get_summary()
    
    print("\n2. Testing Next-Step Predictor")
    print("-" * 50)
    predictor = NextStepPredictor(
        input_shape=(timesteps, features),
        output_steps=5,
        lstm_units=[128, 64],
        output_features=features
    )
    predictor.build()
    predictor.compile()
    predictor.get_summary()
    
    # Test with dummy data
    print("\n3. Testing with dummy data")
    print("-" * 50)
    dummy_input = tf.random.normal((32, timesteps, features))
    
    print("Autoencoder forward pass:")
    ae_output = ae.autoencoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {ae_output.shape}")
    
    print("\nNext-step predictor forward pass:")
    ns_output = predictor.model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {ns_output.shape}")
    
    print("\nEncoder output (latent representation):")
    latent = ae.encoder(dummy_input)
    print(f"Latent shape: {latent.shape}")
