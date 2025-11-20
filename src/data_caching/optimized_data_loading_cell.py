"""
OPTIMIZED DATA LOADING CELL
Add this cell to your notebook AFTER the data loading functions cell 
and BEFORE the training execution cell.

This will cache the preprocessed data so you don't have to process it every time!
"""

# ============================================================================
# OPTION 1: Using the helper module (Recommended)
# ============================================================================

from data_cache_helper import load_or_process_data

# Configure paths
prediction_data_dir = '/kaggle/input/nfl-big-data-bowl-2026-prediction/train'
cache_dir = './cached_training_data'  # Where to store cached arrays

# Set to True to force reprocessing (useful if you changed preprocessing logic)
FORCE_REPROCESS = False

# Load or process data (automatically uses cache if available)
X_train, X_val, y_train, y_val, preprocessor = load_or_process_data(
    data_dir=prediction_data_dir,
    data_loader_func=load_and_prepare_data,
    test_size=0.2,
    random_state=42,
    cache_dir=cache_dir,
    force_reprocess=FORCE_REPROCESS
)

# Create TensorFlow datasets from the arrays
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

print(f"\n✅ Data ready for training!")
print(f"   Training samples: {len(X_train):,}")
print(f"   Validation samples: {len(X_val):,}")


# ============================================================================
# OPTION 2: Manual caching (if you prefer more control)
# ============================================================================

"""
import os
import numpy as np

# Paths for cached files
CACHE_DIR = './cached_training_data'
X_TRAIN_PATH = f'{CACHE_DIR}/X_train.npy'
X_VAL_PATH = f'{CACHE_DIR}/X_val.npy'
Y_TRAIN_PATH = f'{CACHE_DIR}/y_train.npy'
Y_VAL_PATH = f'{CACHE_DIR}/y_val.npy'
PREPROCESSOR_CACHE_PATH = f'{CACHE_DIR}/preprocessor_cache.joblib'

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Check if cache exists
cache_exists = all([
    os.path.exists(X_TRAIN_PATH),
    os.path.exists(X_VAL_PATH),
    os.path.exists(Y_TRAIN_PATH),
    os.path.exists(Y_VAL_PATH),
    os.path.exists(PREPROCESSOR_CACHE_PATH)
])

if cache_exists and not FORCE_REPROCESS:
    print("📂 Loading preprocessed data from cache...")
    X_train = np.load(X_TRAIN_PATH)
    X_val = np.load(X_VAL_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_val = np.load(Y_VAL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_CACHE_PATH)
    
    print(f"✅ Loaded from cache successfully!")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_val shape: {X_val.shape}")
    
else:
    print("🔄 Processing data (this will take time on first run)...")
    
    # Original data loading
    prediction_data_dir = '/kaggle/input/nfl-big-data-bowl-2026-prediction/train'
    train_ds, val_ds, preprocessor = load_and_prepare_data(prediction_data_dir)
    
    # Convert datasets to numpy arrays
    print("📊 Converting to numpy arrays for caching...")
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    
    for features, labels in train_ds:
        X_train_list.append(features.numpy())
        y_train_list.append(labels.numpy())
    
    for features, labels in val_ds:
        X_val_list.append(features.numpy())
        y_val_list.append(labels.numpy())
    
    X_train = np.vstack(X_train_list)
    y_train = np.vstack(y_train_list)
    X_val = np.vstack(X_val_list)
    y_val = np.vstack(y_val_list)
    
    # Save to cache
    print("💾 Saving to cache for future runs...")
    np.save(X_TRAIN_PATH, X_train)
    np.save(X_VAL_PATH, X_val)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_VAL_PATH, y_val)
    joblib.dump(preprocessor, PREPROCESSOR_CACHE_PATH)
    
    print(f"✅ Cache created successfully!")

# Create TensorFlow datasets from the arrays
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

print(f"\n✅ Data ready for training!")
print(f"   Training samples: {len(X_train):,}")
print(f"   Validation samples: {len(X_val):,}")
"""
