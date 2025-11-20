#!/usr/bin/env python3
"""
Test script for the data caching system.
Run this to verify the caching functionality works correctly.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_cache_helper import DataCache
import numpy as np


def test_cache_system():
    """Test the data cache save/load functionality."""
    
    print("=" * 80)
    print("Testing Data Cache System")
    print("=" * 80)
    
    # Create test cache
    cache_dir = './test_cache'
    cache = DataCache(cache_dir)
    
    print("\n1️⃣  Testing cache detection (should be False initially)...")
    assert not cache.exists(), "Cache should not exist yet"
    print("   ✅ Pass: Cache correctly reported as not existing")
    
    print("\n2️⃣  Creating mock preprocessed data...")
    # Create mock data
    X_train = np.random.rand(1000, 10, 50)  # 1000 samples, 10 timesteps, 50 features
    X_val = np.random.rand(200, 10, 50)
    y_train = np.random.rand(1000, 2)  # 1000 samples, 2 outputs (x, y)
    y_val = np.random.rand(200, 2)
    
    # Mock preprocessor (just a dict for testing)
    preprocessor = {
        'test_key': 'test_value',
        'features': ['feature1', 'feature2']
    }
    
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_val shape: {X_val.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   y_val shape: {y_val.shape}")
    print("   ✅ Pass: Mock data created")
    
    print("\n3️⃣  Saving data to cache...")
    metadata = {
        'test_run': True,
        'sample_count': 1000
    }
    cache.save(X_train, X_val, y_train, y_val, preprocessor, metadata)
    print("   ✅ Pass: Data saved successfully")
    
    print("\n4️⃣  Testing cache detection (should be True now)...")
    assert cache.exists(), "Cache should exist after saving"
    print("   ✅ Pass: Cache correctly detected")
    
    print("\n5️⃣  Displaying cache info...")
    cache.info()
    print("   ✅ Pass: Cache info displayed")
    
    print("\n6️⃣  Loading data from cache...")
    X_train_loaded, X_val_loaded, y_train_loaded, y_val_loaded, preprocessor_loaded = cache.load()
    print("   ✅ Pass: Data loaded successfully")
    
    print("\n7️⃣  Verifying loaded data matches original...")
    assert np.array_equal(X_train, X_train_loaded), "X_train mismatch"
    assert np.array_equal(X_val, X_val_loaded), "X_val mismatch"
    assert np.array_equal(y_train, y_train_loaded), "y_train mismatch"
    assert np.array_equal(y_val, y_val_loaded), "y_val mismatch"
    assert preprocessor == preprocessor_loaded, "Preprocessor mismatch"
    print("   ✅ Pass: All data matches perfectly!")
    
    print("\n8️⃣  Testing cache clearing...")
    cache.clear()
    assert not cache.exists(), "Cache should not exist after clearing"
    print("   ✅ Pass: Cache cleared successfully")
    
    print("\n" + "=" * 80)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe caching system is working correctly.")
    print("You can now use it in your notebook to save time and resources!")
    

if __name__ == '__main__':
    try:
        test_cache_system()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
