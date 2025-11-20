#!/usr/bin/env python3
"""
Script to add data caching cells to predictor.ipynb
This will insert two new cells after the training execution cell.
"""

import json
import sys
from pathlib import Path

# New markdown cell explaining the caching
cache_markdown = {
    "cell_type": "markdown",
    "id": "cache_explanation",
    "metadata": {},
    "source": [
        "### 4.5 Save Preprocessed Data Cache (OPTIONAL)\\n",
        "\\n",
        "**Run this cell ONCE after the first data processing to save the cache.**\\n",
        "This saves the preprocessed `X_train`, `X_val`, `y_train`, `y_val` arrays to disk.\\n",
        "\\n",
        "**Benefits:**\\n",
        "- ⚡ Loads in 5-10 seconds instead of 3-5 minutes\\n",
        "- 💾 Cache size: ~1-1.5 GB\\n",
        "- 🔄 Reusable across notebook sessions\\n",
        "\\n",
        "**Instructions:**\\n",
        "1. Run the training cell above (cell 4) to get `train_ds`, `val_ds`, `preprocessor`\\n",
        "2. Run this cell to save the cache\\n",
        "3. Download the `/kaggle/working/training_cache/` folder from Kaggle\\n",
        "4. Upload it as a Kaggle dataset\\n",
        "5. Use the loading cell below in future runs"
    ]
}

#Save cache cell
save_cache_cell = {
    "cell_type": "code",
    "id": "save_training_cache",
    "metadata": {"trusted": True},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# ============================================================\\n",
        "# SAVE TRAINING DATA CACHE\\n",
        "# Run this cell ONCE to generate the cache files\\n",
        "# ============================================================\\n",
        "\\n",
        "import os\\n",
        "import numpy as np\\n",
        "import joblib\\n",
        "from datetime import datetime\\n",
        "\\n",
        "# Configuration\\n",
        "CACHE_DIR = '/kaggle/working/training_cache'  # Kaggle working directory\\n",
        "os.makedirs(CACHE_DIR, exist_ok=True)\\n",
        "\\n",
        "print(\\\"💾 Converting TensorFlow datasets to numpy arrays...\\\")\\n",
        "\\n",
        "# Convert train dataset to arrays\\n",
        "X_train_list = []\\n",
        "y_train_list = []\\n",
        "for features, labels in train_ds:\\n",
        "    X_train_list.append(features.numpy())\\n",
        "    y_train_list.append(labels.numpy())\\n",
        "\\n",
        "# Convert validation dataset to arrays\\n",
        "X_val_list = []\\n",
        "y_val_list = []\\n",
        "for features, labels in val_ds:\\n",
        "    X_val_list.append(features.numpy())\\n",
        "    y_val_list.append(labels.numpy())\\n",
        "\\n",
        "# Stack into single arrays\\n",
        "X_train = np.vstack(X_train_list)\\n",
        "y_train = np.vstack(y_train_list)\\n",
        "X_val = np.vstack(X_val_list)\\n",
        "y_val = np.vstack(y_val_list)\\n",
        "\\n",
        "print(f\\\"✅ Conversion complete!\\\")\\n",
        "print(f\\\"   X_train shape: {X_train.shape}\\\")\\n",
        "print(f\\\"   X_val shape: {X_val.shape}\\\")\\n",
        "print(f\\\"   y_train shape: {y_train.shape}\\\")\\n",
        "print(f\\\"   y_val shape: {y_val.shape}\\\")\\n",
        "\\n",
        "# Save arrays\\n",
        "print(f\\\"\\\\n💾 Saving cache to {CACHE_DIR}...\\\")\\n",
        "np.save(f'{CACHE_DIR}/X_train.npy', X_train)\\n",
        "np.save(f'{CACHE_DIR}/X_val.npy', X_val)\\n",
        "np.save(f'{CACHE_DIR}/y_train.npy', y_train)\\n",
        "np.save(f'{CACHE_DIR}/y_val.npy', y_val)\\n",
        "joblib.dump(preprocessor, f'{CACHE_DIR}/preprocessor.joblib')\\n",
        "\\n",
        "# Save metadata\\n",
        "metadata = {\\n",
        "    'created': datetime.now().isoformat(),\\n",
        "    'X_train_shape': X_train.shape,\\n",
        "    'X_val_shape': X_val.shape,\\n",
        "    'y_train_shape': y_train.shape,\\n",
        "    'y_val_shape': y_val.shape,\\n",
        "    'sequence_length': SEQUENCE_LENGTH\\n",
        "}\\n",
        "\\n",
        "with open(f'{CACHE_DIR}/metadata.txt', 'w') as f:\\n",
        "    for key, value in metadata.items():\\n",
        "        f.write(f\\\"{key}: {value}\\\\n\\\")\\n",
        "\\n",
        "# Calculate total size\\n",
        "total_size = sum(\\n",
        "    os.path.getsize(f'{CACHE_DIR}/{fname}') \\n",
        "    for fname in ['X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy', 'preprocessor.joblib']\\n",
        ")\\n",
        "\\n",
        "print(f\\\"\\\\n✅ Cache saved successfully!\\\")\\n",
        "print(f\\\"   Location: {CACHE_DIR}\\\")\\n",
        "print(f\\\"   Total size: {total_size / 1024 / 1024:.2f} MB\\\")\\n",
        "print(f\\\"\\\\n📥 NEXT STEPS:\\\")\\n",
        "print(f\\\"   1. Download the '{CACHE_DIR}' folder from Kaggle\\\")\\n",
        "print(f\\\"   2. Upload it as a new Kaggle dataset\\\")\\n",
        "print(f\\\"   3. Add that dataset to your notebook\\\")\\n",
        "print(f\\\"   4. Use the loading cell below to load from cache\\\")\\n",
        "\\n",
        "# Clean up to free memory\\n",
        "del X_train_list, y_train_list, X_val_list, y_val_list\\n",
        "del X_train, X_val, y_train, y_val"
    ]
}

# Load cache markdown cell
load_cache_markdown = {
    "cell_type": "markdown",
    "id": "load_cache_explanation",
    "metadata": {},
    "source": [
        "### 4.6 Load Preprocessed Data from Cache (OPTIONAL)\\n",
        "\\n",
        "**Use this cell instead of cell 4 to load from cache.**\\n",
        "\\n",
        "**Prerequisites:**\\n",
        "1. You've run the save cache cell above at least once\\n",
        "2. You've uploaded the cache folder as a Kaggle dataset\\n",
        "3. You've added that dataset to this notebook\\n",
        "\\n",
        "**Update the CACHE_INPUT_PATH below to point to your cache dataset!**"
    ]
}

# Load cache cell
load_cache_cell = {
    "cell_type": "code",
    "id": "load_training_cache",
    "metadata": {"trusted": True},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# ============================================================\\n",
        "# LOAD TRAINING DATA FROM CACHE\\n",
        "# Use this cell INSTEAD of the training execution cell (cell 4)\\n",
        "# when you have cached data available\\n",
        "# ============================================================\\n",
        "\\n",
        "import os\\n",
        "import numpy as np\\n",
        "import joblib\\n",
        "import tensorflow as tf\\n",
        "\\n",
        "# IMPORTANT: Update this path to your cache dataset!\\n",
        "# After uploading cache as a dataset, it will be at:\\n",
        "# /kaggle/input/your-cache-dataset-name/training_cache/\\n",
        "CACHE_INPUT_PATH = '/kaggle/input/nfl-training-cache/training_cache'\\n",
        "\\n",
        "# Check if cache exists\\n",
        "if not os.path.exists(CACHE_INPUT_PATH):\\n",
        "    print(f\\\"❌ Cache not found at {CACHE_INPUT_PATH}\\\")\\n",
        "    print(f\\\"   Please update CACHE_INPUT_PATH or run the data processing cell instead.\\\")\\n",
        "    raise FileNotFoundError(f\\\"Cache directory not found: {CACHE_INPUT_PATH}\\\")\\n",
        "\\n",
        "print(f\\\"📂 Loading preprocessed data from cache...\\\")\\n",
        "print(f\\\"   Location: {CACHE_INPUT_PATH}\\\")\\n",
        "\\n",
        "# Load numpy arrays\\n",
        "X_train = np.load(f'{CACHE_INPUT_PATH}/X_train.npy')\\n",
        "X_val = np.load(f'{CACHE_INPUT_PATH}/X_val.npy')\\n",
        "y_train = np.load(f'{CACHE_INPUT_PATH}/y_train.npy')\\n",
        "y_val = np.load(f'{CACHE_INPUT_PATH}/y_val.npy')\\n",
        "preprocessor = joblib.load(f'{CACHE_INPUT_PATH}/preprocessor.joblib')\\n",
        "\\n",
        "print(f\\\"\\\\n✅ Cache loaded successfully!\\\")\\n",
        "print(f\\\"   X_train shape: {X_train.shape}\\\")\\n",
        "print(f\\\"   X_val shape: {X_val.shape}\\\")\\n",
        "print(f\\\"   y_train shape: {y_train.shape}\\\")\\n",
        "print(f\\\"   y_val shape: {y_val.shape}\\\")\\n",
        "\\n",
        "# Create TensorFlow datasets from cached arrays\\n",
        "train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))\\n",
        "val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))\\n",
        "\\n",
        "print(f\\\"\\\\n✅ TensorFlow datasets created!\\\")\\n",
        "print(f\\\"   Training samples: {len(X_train):,}\\\")\\n",
        "print(f\\\"   Validation samples: {len(X_val):,}\\\")\\n",
        "print(f\\\"\\\\n⚡ Ready for model training! Continue to the model building cell.\\\")\\n",
        "\\n",
        "# Continue with rest of training (hardware detection, model building, etc.)\\n",
        "batch_size = 32\\n",
        "epochs = 3\\n",
        "\\n",
        "if train_ds.cardinality().numpy() == 0:\\n",
        "    print(\\\"No training data generated. Please check data loading.\\\")\\n",
        "\\n",
        "# Detect and initialize hardware strategy\\n",
        "tpu_resolver = None\\n",
        "try:\\n",
        "    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()\\n",
        "    print('TPU found with resolver: ', tpu_resolver.master())\\n",
        "except ValueError:\\n",
        "    print(\\\"Could not initialize TPU resolver. Falling back to other checks.\\\")\\n",
        "\\n",
        "if tpu_resolver:\\n",
        "    tf.config.experimental_connect_to_cluster(tpu_resolver)\\n",
        "    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)\\n",
        "    strategy = tf.distribute.TPUStrategy(tpu_resolver)\\n",
        "    print(\\\"Running on TPU\\\")\\n",
        "else:\\n",
        "    gpus = tf.config.list_physical_devices('GPU')\\n",
        "    if len(gpus) > 0:\\n",
        "        strategy = tf.distribute.MirroredStrategy()\\n",
        "        print(f'Running on {len(gpus)} GPU(s).')\\n",
        "    else:\\n",
        "        strategy = tf.distribute.get_strategy()\\n",
        "        print('Running on CPU.')\\n",
        "\\n",
        "print(\\\"REPLICAS: \\\", strategy.num_replicas_in_sync)\\n",
        "\\n",
        "# Get the input and output shapes from the dataset specs\\n",
        "feature_spec, label_spec = train_ds.element_spec\\n",
        "input_features = feature_spec.shape[1]\\n",
        "output_shape = label_spec.shape[0]\\n",
        "\\n",
        "# Build and compile the model within the strategy scope\\n",
        "with strategy.scope():\\n",
        "    model = build_model(input_features, output_shape)\\n",
        "\\n",
        "model.summary()\\n",
        "\\n",
        "train_model(model, train_ds, val_ds, epochs, batch_size)"
    ]
}

def add_cache_cells_to_notebook(notebook_path):
    """Add caching cells to the notebook after the training execution cell."""
    
    print(f"Reading notebook: {notebook_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the training execution cell (cell with id 'd2455736')
    cells = notebook['cells']
    insert_position = None
    
    for i, cell in enumerate(cells):
        if cell.get('id') == 'd2455736':
            insert_position = i + 1
            print(f"Found training execution cell at position {i}")
            break
    
    if insert_position is None:
        print("❌ Could not find training execution cell!")
        return False
    
    # Check if cache cells already exist
    cache_cell_ids = {'cache_explanation', 'save_training_cache', 'load_cache_explanation', 'load_training_cache'}
    existing_ids = {cell.get('id') for cell in cells}
    
    if cache_cell_ids.intersection(existing_ids):
        print("⚠️  Cache cells already exist in notebook!")
        response = input("Do you want to replace them? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return False
        
        # Remove existing cache cells
        cells[:] = [cell for cell in cells if cell.get('id') not in cache_cell_ids]
        
        # Find new insert position
        for i, cell in enumerate(cells):
            if cell.get('id') == 'd2455736':
                insert_position = i + 1
                break
    
    # Insert the new cells
    cells.insert(insert_position, cache_markdown)
    cells.insert(insert_position + 1, save_cache_cell)
    cells.insert(insert_position + 2, load_cache_markdown)
    cells.insert(insert_position + 3, load_cache_cell)
    
    print(f"✅ Added 4 new cells at position {insert_position}")
    
    # Write back the notebook
    backup_path = str(notebook_path).replace('.ipynb', '_backup.ipynb')
    print(f"💾 Creating backup: {backup_path}")
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Notebook updated successfully!")
    print(f"\n📝 New cells added:")
    print(f"   - Cell 4.5 (markdown): Cache explanation")
    print(f"   - Cell 4.5: Save cache code")
    print(f"   - Cell 4.6 (markdown): Load cache explanation")
    print(f"   - Cell 4.6: Load cache code")
    
    return True

if __name__ == '__main__':
    notebook_path = Path(__file__).parent / 'predictor.ipynb'
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        sys.exit(1)
    
    success = add_cache_cells_to_notebook(notebook_path)
    sys.exit(0 if success else 1)
