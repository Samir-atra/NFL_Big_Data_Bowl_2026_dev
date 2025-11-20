# Data Caching Guide for NFL Big Data Bowl 2026

## Problem
The data preprocessing pipeline in `predictor.ipynb` is resource-intensive:
- Takes several minutes to process
- Uses significant RAM
- Needs to be rerun every time you restart the notebook

## Solution
Cache the preprocessed `X_train`, `X_val`, `y_train`, `y_val` arrays to disk after the first run, then load them directly in subsequent runs.

---

## Quick Start

### Step 1: Add the import
In your first cell, add:
```python
from data_cache_helper import load_or_process_data
```

### Step 2: Replace the training execution cell
Replace this original code:
```python
prediction_data_dir = '/kaggle/input/nfl-big-data-bowl-2026-prediction/train'
train_ds, val_ds, preprocessor = load_and_prepare_data(prediction_data_dir)
```

With this optimized version:
```python
# Configure caching
prediction_data_dir = '/kaggle/input/nfl-big-data-bowl-2026-prediction/train'
cache_dir = './cached_training_data'
FORCE_REPROCESS = False  # Set to True to ignore cache and reprocess

# Load or process data (uses cache automatically)
X_train, X_val, y_train, y_val, preprocessor = load_or_process_data(
    data_dir=prediction_data_dir,
    data_loader_func=load_and_prepare_data,
    cache_dir=cache_dir,
    force_reprocess=FORCE_REPROCESS
)

# Create TensorFlow datasets from arrays
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
```

### Step 3: Run the notebook
- **First run**: Will process data normally and save cache (~3-5 minutes)
- **Subsequent runs**: Will load from cache (~5-10 seconds) ⚡

---

## Manual Approach (Alternative)

If you prefer not to use the helper module, you can cache manually:

```python
import os
import numpy as np
import joblib

# Paths
CACHE_DIR = './cached_training_data'
os.makedirs(CACHE_DIR, exist_ok=True)

# Check if cache exists
cache_files = {
    'X_train': f'{CACHE_DIR}/X_train.npy',
    'X_val': f'{CACHE_DIR}/X_val.npy',
    'y_train': f'{CACHE_DIR}/y_train.npy',
    'y_val': f'{CACHE_DIR}/y_val.npy',
    'preprocessor': f'{CACHE_DIR}/preprocessor.joblib'
}

if all(os.path.exists(f) for f in cache_files.values()):
    # Load from cache
    print("Loading from cache...")
    X_train = np.load(cache_files['X_train'])
    X_val = np.load(cache_files['X_val'])
    y_train = np.load(cache_files['y_train'])
    y_val = np.load(cache_files['y_val'])
    preprocessor = joblib.load(cache_files['preprocessor'])
else:
    # Process and save cache
    print("Processing data...")
    train_ds, val_ds, preprocessor = load_and_prepare_data(prediction_data_dir)
    
    # Convert to arrays
    X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []
    
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
    
    # Save cache
    np.save(cache_files['X_train'], X_train)
    np.save(cache_files['X_val'], X_val)
    np.save(cache_files['y_train'], y_train)
    np.save(cache_files['y_val'], y_val)
    joblib.dump(preprocessor, cache_files['preprocessor'])
    print("Cache saved!")

# Create datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
```

---

## Cache Management

### View cache info
```python
from data_cache_helper import DataCache
cache = DataCache('./cached_training_data')
cache.info()
```

### Clear cache (to free disk space)
```python
cache.clear()
```

### Force reprocessing
```python
# When you've modified the preprocessing logic
FORCE_REPROCESS = True
```

---

## File Sizes

Typical cache sizes (approximate):
- `X_train.npy`: ~800-1200 MB
- `X_val.npy`: ~200-300 MB
- `y_train.npy`: ~2-5 MB
- `y_val.npy`: ~0.5-1 MB
- `preprocessor.joblib`: ~1-2 MB
- **Total**: ~1-1.5 GB

Make sure you have sufficient disk space!

---

## Benefits

✅ **Time Savings**: 3-5 minutes → 5-10 seconds (30-60x faster)  
✅ **RAM Efficiency**: Only load what you need  
✅ **Reproducibility**: Same train/val split every time  
✅ **Iteration Speed**: Quickly experiment with model architectures  

---

## When to Reprocess (Set `FORCE_REPROCESS = True`)

- Changed preprocessing logic
- Modified feature engineering
- Updated `SEQUENCE_LENGTH`
- Changed `train_test_split` parameters
- Using different source data

---

## Troubleshooting

### "Cache not found" error
- Run the notebook once without cache to generate it
- Check that `cache_dir` path is correct

### Out of disk space
- Clear cache: `cache.clear()`
- Use external storage for cache directory

### Stale cache (wrong data)
- Delete cache folder manually or use `cache.clear()`
- Set `FORCE_REPROCESS = True` and run again

---

## Notes

- Cache is stored locally and won't sync across machines
- On Kaggle notebooks, cache is session-specific
- Cache files use NumPy's `.npy` format (efficient and fast)
- Preprocessor is saved with joblib for sklearn compatibility
