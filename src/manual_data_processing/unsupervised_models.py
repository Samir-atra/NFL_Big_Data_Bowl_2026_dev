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
        """Initialize the LSTM Autoencoder.
        
        Args:
            input_shape: Shape of input (timesteps, features)
            latent_dim: Dimension of latent representation
            lstm_units: List of LSTM units for encoder layers
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        
    def build_encoder(self):
        """Build the encoder network."""
        inputs = layers.Input(shape=self.input_shape, name='encoder_input')
        
        x = inputs
        # Stack LSTM layers
        for i, units in enumerate(self.lstm_units[:-1]):
            x = layers.LSTM(
                units, 
                return_sequences=True,
                name=f'encoder_lstm_{i+1}'
            )(x)
            x = layers.Dropout(0.2)(x)
        
        # Last LSTM layer doesn't return sequences
        x = layers.LSTM(
            self.lstm_units[-1],
            return_sequences=False,
            name=f'encoder_lstm_{len(self.lstm_units)}'
        )(x)
        x = layers.Dropout(0.2)(x)
        
        # Latent representation
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        self.encoder = Model(inputs, latent, name='encoder')
        return self.encoder
    
    def build_decoder(self):
        """Build the decoder network."""
        # Decoder input is the latent vector
        latent_inputs = layers.Input(shape=(self.latent_dim,), name='decoder_input')
        
        # Repeat the latent vector for each timestep
        x = layers.RepeatVector(self.input_shape[0])(latent_inputs)
        
        # Stack LSTM layers in reverse
        for i, units in enumerate(reversed(self.lstm_units)):
            x = layers.LSTM(
                units,
                return_sequences=True,
                name=f'decoder_lstm_{i+1}'
            )(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer to reconstruct features
        outputs = layers.TimeDistributed(
            layers.Dense(self.input_shape[1], activation='linear'),
            name='reconstruction'
        )(x)
        
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        return self.decoder
    
    def build_autoencoder(self):
        """Build the complete autoencoder."""
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
        """Compile the autoencoder."""
        if self.autoencoder is None:
            self.build_autoencoder()
        
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
    def get_summary(self):
        """Print model summaries."""
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
    
    Predicts future timesteps given past timesteps, which can be used
    as a pre-training task for the supervised trajectory prediction.
    """
    
    def __init__(self, input_shape, output_steps=5, lstm_units=[256, 128], output_features=None):
        """Initialize the next-step predictor.
        
        Args:
            input_shape: Shape of input (timesteps, features)
            output_steps: Number of future steps to predict
            lstm_units: List of LSTM units
            output_features: Number of output features (if None, same as input features)
        """
        self.input_shape = input_shape
        self.output_steps = output_steps
        self.lstm_units = lstm_units
        self.output_features = output_features or input_shape[1]
        self.model = None
        
    def build(self):
        """Build the next-step prediction model."""
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        x = inputs
        # Stack LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_seq = (i < len(self.lstm_units) - 1)
            x = layers.LSTM(
                units,
                return_sequences=return_seq,
                name=f'lstm_{i+1}'
            )(x)
            x = layers.Dropout(0.2)(x)
        
        # Prediction head
        # Expand to output_steps timesteps
        x = layers.RepeatVector(self.output_steps)(x)
        x = layers.LSTM(128, return_sequences=True, name='prediction_lstm')(x)
        
        # Output for each timestep
        outputs = layers.TimeDistributed(
            layers.Dense(self.output_features, activation='linear'),
            name='predictions'
        )(x)
        
        self.model = Model(inputs, outputs, name='next_step_predictor')
        return self.model
    
    def compile(self, learning_rate=0.001):
        """Compile the model."""
        if self.model is None:
            self.build()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def get_summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()


def create_training_callbacks(model_path, patience=10):
    """Create standard callbacks for training.
    
    Args:
        model_path: Path to save best model
        patience: Patience for early stopping
        
    Returns:
        List of callbacks
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
    """Transfer weights from pretrained encoder to supervised model.
    
    Args:
        pretrained_encoder: The pretrained encoder model
        supervised_model: The supervised model to transfer weights to
        freeze_encoder: Whether to freeze the transferred weights
        
    Returns:
        The supervised model with transferred weights
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
