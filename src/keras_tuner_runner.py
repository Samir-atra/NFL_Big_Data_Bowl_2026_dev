
"""
This module orchestrates the hyperparameter tuning process for the NFL Big Data Bowl 2026 prediction model using Keras Tuner.

It defines the model architecture, specifies the search space for hyperparameters, and configures the tuning process.
The script loads the dataset, creates TensorFlow datasets for training and validation, and then initiates the hyperparameter search.

Key functionalities include:
- **Model Building (`build_model`):** Defines a Keras LSTM model with tunable hyperparameters (learning rate, LSTM units, regularization, activation functions).
- **Hyperparameter Experimentation (`experimenting`):** Uses Keras Tuner's `RandomSearch` to find the best combination of hyperparameters by training and evaluating models.
- **Data Loading and Preparation:** Utilizes `NFLDataLoader` and `create_tf_datasets` from `csv_to_numpy` to load and prepare data.
- **Main Execution Block:** Sets up configuration, loads data, creates datasets, and runs the hyperparameter tuning experiment.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
import keras_tuner

# --- Path Configuration ---
# Add the manual_data_processing directory to the Python path to allow importing custom modules.
# This is crucial for accessing helper classes like NFLDataLoader and create_tf_datasets.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'manual_data_processing'))

# Import necessary data handling utilities from the custom module.
from csv_to_numpy import NFLDataLoader, create_tf_datasets

# --- Global Variables for Model Shape ---
# These will be determined dynamically based on the data.
input_seq_length = None
input_features = None
output_seq_length = None
output_features = None

def build_model(hp: keras_tuner.HyperParameters) -> keras.Model:
    """
    Builds and compiles a Keras LSTM model with a hyperparameter search space.

    This function defines the architecture for a sequence-to-sequence LSTM model.
    It uses Keras Tuner's `HyperParameters` object (`hp`) to specify ranges and
    choices for various hyperparameters, allowing Keras Tuner to search for the
    optimal configuration.

    Args:
        hp (keras_tuner.HyperParameters): An object to define the tunable parameters
                                          of the model.

    Returns:
        keras.Model: A compiled Keras model ready for training or hyperparameter tuning.
    """
    
    SEED = 42 # Fixed random seed for reproducibility in model initialization.
    
    # --- Define Hyperparameter Search Space ---
    # Learning rate for the Adam optimizer.
    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-3, sampling="log")
    # Number of units in the LSTM layers.
    layer_u = hp.Int("lu", min_value=160, max_value=1024, step=8)
    # L2 kernel regularization strength.
    kernel_r = hp.Float("kr", min_value=1e-10, max_value=1e-5, sampling="log")
    # Activation function for LSTM layers.
    acti_f = hp.Choice("af", ["sigmoid", "hard_sigmoid", "tanh"])
    # Weight decay (L2 regularization) for the optimizer. Note: Commented out as Adam does not directly accept weight_decay in this form.
    weight_d = hp.Float("wd", min_value=1e-10, max_value=0.0009, sampling="log")

    # --- Model Architecture ---
    # Using Keras Sequential API for a linear stack of layers.
    model = keras.Sequential([
        # Input layer: Defines the expected shape of the input data.
        # `input_seq_length` and `input_features` are expected to be set globally before this function is called.
        keras.layers.Input(shape=(input_seq_length, input_features)),
        
        # Encoder LSTM layer(s)
        # This is the primary layer for sequence processing.
        keras.layers.LSTM(
            units=layer_u,             # Number of LSTM units, defined by hyperparameter 'lu'
            activation=acti_f,         # Activation function, defined by hyperparameter 'af'
            return_sequences=True,     # Return output for each time step, required for stacked LSTMs or TimeDistributed layers
            kernel_regularizer=keras.regularizers.L2(l2=kernel_r), # L2 regularization on kernel weights, defined by 'kr'
            seed=SEED,                 # Seed for weight initialization randomness
        ),
        # Optional: Add more LSTM layers if needed. Uncomment to enable.
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
        
        # Lambda layer to crop the output sequence if necessary.
        # This ensures the output sequence length matches `output_seq_length`.
        layers.Lambda(lambda x: x[:, :output_seq_length, :]),
        
        # TimeDistributed Dense layer for outputting predictions for each time step.
        keras.layers.TimeDistributed(
            keras.layers.Dense(units=output_features, activation="linear") # Output layer with linear activation for regression
        ),
    ])

    # --- Model Compilation ---
    # Compile the model with the Adam optimizer and Mean Squared Error loss.
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate, # Tunable learning rate
            global_clipnorm=1,           # Gradient clipping to prevent exploding gradients
            amsgrad=False,               # Whether to use the AMSGrad variant of Adam
            # weight_decay=weight_d,     # Tunable weight decay (commented out, not directly used by Adam optimizer here)
        ),
        metrics=[tf.keras.metrics.MeanAbsoluteError()], # Track Mean Absolute Error during training/evaluation
    )

    return model

def experimenting(training_dataset, validation_data) -> None:
    """
    Configures and runs hyperparameter tuning experiments using Keras Tuner's RandomSearch.

    This function sets up the `RandomSearch` tuner, defines the search objective
    (minimizing validation loss), and then initiates the search process.
    It prints summaries of the search space and the final results.

    Args:
        training_dataset: A `tf.data.Dataset` or Keras Sequence object for training data.
        validation_data: A `tf.data.Dataset` or Keras Sequence object for validation data.
    """

    # Initialize HyperParameters object for defining the search space.
    hp = keras_tuner.HyperParameters()
    
    # --- Determine Data Shapes ---
    # Keras Tuner needs to know the input/output shapes. We get this from the first batch.
    # This is crucial for defining the Input layer and subsequent operations.
    try:
        # Get a sample batch from the training dataset.
        x_batch, y_batch = training_dataset[0]
        
        # Declare global variables to store detected shapes.
        global input_features, input_seq_length, output_seq_length, output_features
        
        # Extract shapes from the batch.
        input_seq_length = x_batch.shape[1] # Sequence length of input features
        input_features = x_batch.shape[2]   # Number of features per timestep in input
        output_seq_length = y_batch.shape[1] # Sequence length of output features
        output_features = y_batch.shape[2]  # Number of features per timestep in output
        
        print(f"\nDetected shapes:")
        print(f"  Input: ({input_seq_length}, {input_features})")
        print(f"  Output: ({output_seq_length}, {output_features})")
    except Exception as e:
        print(f"Error detecting data shapes: {e}. Ensure training_dataset is not empty and has correct format.")
        return

    # Instantiate a dummy model to build the search space based on the detected shapes.
    # This call is necessary for `tuner.search_space_summary()` to work correctly.
    build_model(hp)

    # --- Keras Tuner Configuration ---
    # Initialize Keras Tuner's RandomSearch algorithm.
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model, # Function to build the model
        max_trials=10,          # Maximum number of hyperparameter combinations to test.
        objective=keras_tuner.Objective("val_loss", "min"), # Target: minimize validation loss.
        executions_per_trial=1, # Number of models to train per trial (1 for faster tuning).
        overwrite=True,         # Overwrite previous tuning results in the directory.
        # Directory to save tuner logs and models. Defaults to './tuner_results' if env var is not set.
        directory=os.getenv("KERAS_TUNER_EXPERIMENTS_DIR", "./tuner_results"), 
        project_name="nfl_prediction", # Subdirectory name for this project's results.
    )

    # Print a summary of the hyperparameter search space defined in build_model.
    tuner.search_space_summary()

    # --- Running the Search ---
    # The `training_dataset` is already batched by `create_tf_datasets`, so no further batching is needed.
    print("\nStarting hyperparameter search...")
    try:
        tuner.search(
            training_dataset,          # The training data
            validation_data=validation_data, # The validation data
            epochs=2                   # Number of epochs for each trial model training.
                                       # Note: A small number of epochs is used here for a quick demonstration.
                                       # For actual tuning, this should be higher or use callbacks like EarlyStopping.
        )
    except Exception as e:
        print(f"Error during Keras Tuner search: {e}")
        return

    # Print a summary of the best performing trials and their hyperparameters.
    print("\nHyperparameter search complete.")
    tuner.results_summary()


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration Parameters ---
    # Directory containing the training data.
    train_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    # Batch size for creating TensorFlow datasets.
    batch_size = 32
    # Number of epochs for training during each hyperparameter trial.
    # (Note: This is set to a low value for demonstration; adjust for actual tuning).
    epochs = 50 # This variable is not directly used in tuner.search, but is kept for context.
    # Proportion of data to be used for validation.
    test_size = 0.2
    
    print("\n" + "="*60)
    print("NFL Big Data Bowl 2026 - Predictor Training and Hyperparameter Tuning")
    print("="*60)
    
    # --- Data Loading and Preparation ---
    print("\n[1/4] Loading data from CSV files...")
    # Initialize the data loader with the training directory.
    loader = NFLDataLoader(train_dir)
    # Load aligned feature (X) and label (y) data.
    X, y = loader.get_aligned_data()
    
    # Check if data was loaded successfully.
    if len(X) == 0:
        print("Error: No data loaded. Please check the data directory and ensure CSV files exist.")
    else:
        # Print basic information about the loaded data.
        print(f"\nData Summary:")
        print(f"  Total sequences: {len(X)}")
        # Safely access sequence lengths and feature counts, handling cases where X or y might be empty.
        if X:
            print(f"  Sample input sequence length: {len(X[0]) if X[0] else 0}")
            print(f"  Input features per timestep: {len(X[0][0]) if len(X[0]) > 0 and X[0][0] else 0}")
        if y:
            print(f"  Sample output sequence length: {len(y[0]) if y[0] else 0}")
            print(f"  Output features per timestep: {len(y[0][0]) if len(y[0]) > 0 and y[0][0] else 0}")
    
        # Create TensorFlow datasets for training and validation.
        # This function handles splitting, padding, and batching.
        print(f"\n[2/4] Creating training and validation sequences (test_size={test_size}, batch_size={batch_size})...")
        train_seq, val_seq = create_tf_datasets(X, y, test_size=test_size, batch_size=batch_size)

        # Ensure datasets were created before proceeding.
        if train_seq is None or val_seq is None:
            print("Error: Failed to create training or validation sequences.")
        else:
            # --- Hyperparameter Experimentation ---
            print("\n[3/4] Initiating hyperparameter experimentation...")
            # Call the `experimenting` function to run the Keras Tuner search.
            experimenting(train_seq, val_seq)
            
            print("\n[4/4] Hyperparameter tuning process initiated.")
            print("Check './tuner_results/nfl_prediction' directory for experiment logs and best models.")

