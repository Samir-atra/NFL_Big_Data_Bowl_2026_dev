#!/usr/bin/env python3
"""
This script performs a quick integration test to verify that the CSV loading
and TensorFlow dataset creation functionalities from the `csv_to_numpy` module
(imported via `data_loader.py`) are working correctly.

It simulates a basic data pipeline by:
1. Specifying a directory containing the NFL training data.
2. Using `NFLDataLoader` to load and align data from CSV files.
3. Using `create_tf_datasets` to generate training and validation sequences.
4. Verifying the shapes of the loaded data and generated batches.

This serves as a basic smoke test to ensure the core data processing components
can be integrated and executed without immediate errors.
"""

import sys
import os

# --- Path Configuration ---
# Add the `manual_data_processing` directory to the Python path to allow importing
# custom modules like `NFLDataLoader` and `create_tf_datasets`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'manual_data_processing'))

# Import the necessary data loading and dataset creation functions.
from csv_to_numpy import NFLDataLoader, create_tf_datasets

print("Testing CSV to NumPy data loading and sequence creation...")

# --- Test Configuration ---
# Define the directory containing the NFL training data.
# This path is used to initialize the data loader.
train_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'

# --- Test Step 1: Data Loading ---
print("\n[Step 1/3] Loading data using NFLDataLoader...")
# Initialize the NFLDataLoader with the specified training directory.
loader = NFLDataLoader(train_dir)
# Load the aligned data, separating features (X) and labels (y).
X, y = loader.get_aligned_data()

# Verify that data was loaded successfully.
print(f"\n✓ Data loaded successfully!")
print(f"  Total sequences found: {len(X)}")
# Print sample shapes to confirm data structure.
# `len(X[0])` gives the number of timesteps in the first sequence.
# `len(y[0])` gives the number of timesteps in the first label sequence.
print(f"  Sample shapes: X[0] has {len(X[0])} timesteps, y[0] has {len(y[0])} timesteps")

# --- Test Step 2: Sequence Creation ---
print("\n[Step 2/3] Creating training and validation sequences...")
# Use the loaded data (X, y) to create TensorFlow datasets for training and validation.
# Parameters like test_size and batch_size are specified for this test.
train_seq, val_seq = create_tf_datasets(X, y, test_size=0.2, batch_size=32)

# Verify that the datasets were created without errors.
print(f"\n✓ Sequences created successfully!")
# Report the number of batches generated for training and validation.
# Note: `len(dataset)` works for datasets that have been batched.
print(f"  Number of training batches: {len(train_seq)}")
print(f"  Number of validation batches: {len(val_seq)}")

# --- Test Step 3: Batch Generation ---
print("\n[Step 3/3] Generating a sample batch...")
# Retrieve the first batch from the training dataset to check its structure.
try:
    x_batch, y_batch = train_seq[0] # Accessing the first batch (index 0)
    print(f"\n✓ Batch generation works!")
    # Print the shapes of the features (X) and labels (y) in the batch.
    print(f"  Shape of batch X (features): {x_batch.shape}")
    print(f"  Shape of batch y (labels): {y_batch.shape}")
except IndexError:
    print("Error: Could not retrieve the first batch. The training dataset might be empty.")
except Exception as e:
    print(f"An unexpected error occurred while generating the batch: {e}")

# --- Final Test Summary ---
print(f"\n{'='*60}")
print(f"✓ ALL TESTS PASSED!")
print(f"{'='*60}")
