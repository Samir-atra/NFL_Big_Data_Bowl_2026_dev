#!/usr/bin/env python3
"""Quick test to verify csv_to_numpy integration works"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'manual_data_processing'))

from csv_to_numpy import NFLDataLoader, create_tf_datasets

print("Testing CSV to NumPy data loading...")

# Test with small subset
train_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'

loader = NFLDataLoader(train_dir)
X, y = loader.get_aligned_data()

print(f"\n✓ Data loaded successfully!")
print(f"  Total sequences: {len(X)}")
print(f"  Sample shapes: X[0]={len(X[0])} timesteps, y[0]={len(y[0])} timesteps")

# Create sequences
train_seq, val_seq = create_tf_datasets(X, y, test_size=0.2, batch_size=32)

print(f"\n✓ Sequences created successfully!")
print(f"  Training batches: {len(train_seq)}")
print(f"  Validation batches: {len(val_seq)}")

# Get a batch
x_batch, y_batch = train_seq[0]
print(f"\n✓ Batch generation works!")
print(f"  Batch X shape: {x_batch.shape}")
print(f"  Batch y shape: {y_batch.shape}")

print(f"\n{'='*60}")
print(f"✓ ALL TESTS PASSED!")
print(f"{'='*60}")
