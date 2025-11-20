"""
Helper module for caching preprocessed training data.
This saves time and RAM by avoiding reprocessing data on every run.
"""

import os
import numpy as np
import joblib
from pathlib import Path


class DataCache:
    """Manages saving and loading of preprocessed training data."""
    
    def __init__(self, cache_dir='./cached_data'):
        """
        Initialize the DataCache.
        
        Args:
            cache_dir (str): Directory where cache files will be stored.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Define cache file paths
        self.X_train_path = self.cache_dir / 'X_train.npy'
        self.X_val_path = self.cache_dir / 'X_val.npy'
        self.y_train_path = self.cache_dir / 'y_train.npy'
        self.y_val_path = self.cache_dir / 'y_val.npy'
        self.preprocessor_path = self.cache_dir / 'preprocessor_cache.joblib'
        self.metadata_path = self.cache_dir / 'cache_metadata.txt'
    
    def exists(self):
        """
        Check if all required cache files exist.
        
        Returns:
            bool: True if all cache files exist, False otherwise.
        """
        required_files = [
            self.X_train_path,
            self.X_val_path,
            self.y_train_path,
            self.y_val_path,
            self.preprocessor_path
        ]
        return all(f.exists() for f in required_files)
    
    def save(self, X_train, X_val, y_train, y_val, preprocessor, metadata=None):
        """
        Save preprocessed data and preprocessor to cache.
        
        Args:
            X_train (np.ndarray): Training features.
            X_val (np.ndarray): Validation features.
            y_train (np.ndarray): Training labels.
            y_val (np.ndarray): Validation labels.
            preprocessor: Fitted preprocessor object.
            metadata (dict, optional): Additional metadata to save.
        """
        print(f"💾 Saving preprocessed data to {self.cache_dir}...")
        
        # Save numpy arrays (uses compression for smaller file size)
        np.save(self.X_train_path, X_train)
        np.save(self.X_val_path, X_val)
        np.save(self.y_train_path, y_train)
        np.save(self.y_val_path, y_val)
        
        # Save preprocessor
        joblib.dump(preprocessor, self.preprocessor_path)
        
        # Save metadata
        if metadata:
            with open(self.metadata_path, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
        
        # Print file sizes
        total_size = sum(
            os.path.getsize(p) for p in [
                self.X_train_path, self.X_val_path,
                self.y_train_path, self.y_val_path,
                self.preprocessor_path
            ]
        )
        
        print(f"✅ Cache saved successfully!")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   X_val shape: {X_val.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   y_val shape: {y_val.shape}")
        print(f"   Total cache size: {total_size / 1024 / 1024:.2f} MB")
    
    def load(self):
        """
        Load preprocessed data and preprocessor from cache.
        
        Returns:
            tuple: (X_train, X_val, y_train, y_val, preprocessor)
        
        Raises:
            FileNotFoundError: If cache files don't exist.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"Cache not found in {self.cache_dir}. "
                "Please run data processing first to create the cache."
            )
        
        print(f"📂 Loading preprocessed data from {self.cache_dir}...")
        
        # Load numpy arrays
        X_train = np.load(self.X_train_path)
        X_val = np.load(self.X_val_path)
        y_train = np.load(self.y_train_path)
        y_val = np.load(self.y_val_path)
        
        # Load preprocessor
        preprocessor = joblib.load(self.preprocessor_path)
        
        print(f"✅ Cache loaded successfully!")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   X_val shape: {X_val.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   y_val shape: {y_val.shape}")
        
        return X_train, X_val, y_train, y_val, preprocessor
    
    def clear(self):
        """Delete all cache files."""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            print(f"🗑️  Cache cleared from {self.cache_dir}")
    
    def info(self):
        """Display information about the cache."""
        if not self.exists():
            print(f"❌ No cache found in {self.cache_dir}")
            return
        
        print(f"📊 Cache Information:")
        print(f"   Location: {self.cache_dir}")
        
        # Calculate total size
        total_size = 0
        for file_path in [self.X_train_path, self.X_val_path, 
                          self.y_train_path, self.y_val_path,
                          self.preprocessor_path]:
            if file_path.exists():
                size = os.path.getsize(file_path)
                total_size += size
                print(f"   {file_path.name}: {size / 1024 / 1024:.2f} MB")
        
        print(f"   Total: {total_size / 1024 / 1024:.2f} MB")
        
        # Show metadata if exists
        if self.metadata_path.exists():
            print(f"\n   Metadata:")
            with open(self.metadata_path, 'r') as f:
                for line in f:
                    print(f"     {line.strip()}")


def load_or_process_data(data_dir, data_loader_func, test_size=0.2, random_state=42, 
                         cache_dir='./cached_data', force_reprocess=False):
    """
    Load preprocessed data from cache if available, otherwise process and cache it.
    
    Args:
        data_dir (str): Directory containing raw training data.
        data_loader_func (callable): Function to process raw data.
                                     Should return (train_dataset, val_dataset, preprocessor)
        test_size (float): Proportion for validation split.
        random_state (int): Random seed for reproducibility.
        cache_dir (str): Directory for cache files.
        force_reprocess (bool): If True, ignore cache and reprocess data.
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val, preprocessor)
    """
    cache = DataCache(cache_dir)
    
    # Try to load from cache
    if not force_reprocess and cache.exists():
        print("🚀 Found cached preprocessed data!")
        return cache.load()
    
    # Cache doesn't exist or forced reprocessing
    if force_reprocess:
        print("🔄 Force reprocessing data (ignoring cache)...")
    else:
        print("🔄 No cache found. Processing data for the first time...")
    
    # Process the data using the provided function
    # This function should return datasets but we'll extract the arrays
    from datetime import datetime
    start_time = datetime.now()
    
    # We need to modify the data loader to return arrays instead of datasets
    # For now, we'll call the original function and extract arrays
    print("⏳ This may take a while...")
    
    # Import the necessary function - this assumes it's available in the notebook
    import tensorflow as tf
    
    # Call the data loading function
    train_ds, val_ds, preprocessor = data_loader_func(data_dir, test_size, random_state)
    
    # Convert TensorFlow datasets to numpy arrays
    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []
    
    print("📊 Converting datasets to arrays...")
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
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Save to cache
    metadata = {
        'processing_time_seconds': processing_time,
        'processed_date': datetime.now().isoformat(),
        'test_size': test_size,
        'random_state': random_state
    }
    
    cache.save(X_train, X_val, y_train, y_val, preprocessor, metadata)
    print(f"⏱️  Processing took {processing_time:.1f} seconds")
    
    return X_train, X_val, y_train, y_val, preprocessor
