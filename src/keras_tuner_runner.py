import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
import keras_tuner

# Add the manual_data_processing directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'manual_data_processing'))

from csv_to_numpy import NFLDataLoader, create_tf_datasets


def build_model(hp):
    """
    Builds a compiled Keras LSTM model with hyperparameters to be experimented on.

    This function defines the architecture of the LSTM model for sequence-to-sequence prediction.
    It incorporates hyperparameter search spaces for key model parameters like learning rate,
    number of LSTM units, kernel regularization, and activation functions.

    Args:
        hp (keras_tuner.HyperParameters): An instance of Keras Tuner's HyperParameters class,
                                          used to define the search space for hyperparameters.

    Returns:
        keras.Model: The compiled Keras LSTM model with hyperparameters set by Keras Tuner.
    """
    
    SEED = 42
    # Define hyperparameter search spaces for tuning
    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-3, sampling="log")
    layer_u = hp.Int("lu", min_value=160, max_value=1024, step=8)
    kernel_r = hp.Float("kr", min_value=1e-10, max_value=1e-5, sampling="log")
    acti_f = hp.Choice("af", ["sigmoid", "hard_sigmoid", "tanh"])
    weight_d = hp.Float("wd", min_value=1e-10, max_value=0.0009, sampling="log")

    # Define the model structure using Keras Sequential API
    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=(input_seq_length, input_features)),
        
        # Encoder LSTM layers
        keras.layers.LSTM(
            units=layer_u,
            activation=acti_f,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
            seed=SEED,
        ),
        # keras.layers.LSTM(
        #     units=layer_u // 2,
        #     activation=acti_f,
        #     return_sequences=True,
        #     kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
        #     seed=SEED,
        # ),
        # keras.layers.LSTM(
        #     units=layer_u // 2,
        #     activation=acti_f,
        #     return_sequences=True,
        #     kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
        #     seed=SEED,
        # ),
        
        # Crop to output sequence length
        layers.Lambda(lambda x: x[:, :output_seq_length, :]),
        
        # TimeDistributed output layer
        keras.layers.TimeDistributed(
            keras.layers.Dense(units=output_features, activation="linear")
        ),
    ])

    # Compile the model with a tunable optimizer and metrics
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            global_clipnorm=1,
            amsgrad=False,
            # weight_decay=weight_d, # Tunable weight decay
        ),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return model


def experimenting(training_dataset, validation_data):
    """
    Runs Keras Tuner experiments for the LSTM model using the RandomSearch algorithm.

    This function initializes a `RandomSearch` tuner with the `build_model` function,
    configures the search objective (minimizing validation loss), and then executes
    the hyperparameter search across the defined search spaces. It prints summaries
    of the search space and the results.

    Args:
        training_dataset: NFLDataSequence object for training data
        validation_data: NFLDataSequence object for validation data

    """

    hp = keras_tuner.HyperParameters()
    
    # Get a batch from the sequence to determine shapes
    x_batch, y_batch = training_dataset[0]
    global input_features, input_seq_length, output_seq_length, output_features
    input_seq_length = x_batch.shape[1]
    input_features = x_batch.shape[2]
    output_seq_length = y_batch.shape[1]
    output_features = y_batch.shape[2]
    
    print(f"\nDetected shapes:")
    print(f"  Input: ({input_seq_length}, {input_features})")
    print(f"  Output: ({output_seq_length}, {output_features})")
    
    build_model(hp) # Instantiate a dummy model to build the search space

    # Initialize Keras Tuner's RandomSearch algorithm
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        max_trials=10, # Maximum number of hyperparameter combinations to try
        objective=keras_tuner.Objective("val_loss", "min"),   # Objective is to minimize validation loss
        executions_per_trial=1, # Number of models to train for each trial (1 for efficiency)
        overwrite=True, # Overwrite previous results in the directory
        directory=os.getenv("KERAS_TUNER_EXPERIMENTS_DIR", "./tuner_results"), # Directory to save experiment logs and checkpoints
        project_name="nfl_prediction", # Name of the Keras Tuner project
    )

    tuner.search_space_summary() # Print a summary of the hyperparameter search space

    # NFLDataSequence is already batched, no need to call batch() again
    # Run the hyperparameter search experiments
    tuner.search(
        training_dataset, 
        validation_data=validation_data, 
        epochs=2
    )

    tuner.results_summary() # Print a summary of the best performing trials


if __name__ == "__main__":
    train_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    batch_size = 32
    epochs = 50
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
    
    print(f"\nData Summary:")
    print(f"  Total sequences: {len(X)}")
    print(f"  Sample input sequence length: {len(X[0])}")
    print(f"  Sample output sequence length: {len(y[0])}")
    print(f"  Input features per timestep: {len(X[0][0]) if len(X[0]) > 0 else 0}")
    print(f"  Output features per timestep: {len(y[0][0]) if len(y[0]) > 0 else 0}")
    
    # Create Keras Sequences with padding
    print(f"\n[2/4] Creating training and validation sequences (test_size={test_size})...")
    train_seq, val_seq = create_tf_datasets(X, y, test_size=test_size, batch_size=batch_size)
    
    # Run the hyperparameter experimentation
    experimenting(train_seq, val_seq)
