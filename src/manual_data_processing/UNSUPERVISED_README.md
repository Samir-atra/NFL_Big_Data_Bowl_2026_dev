# Unsupervised Pre-training Pipeline for NFL Player Trajectory Prediction

This module provides unsupervised learning capabilities to leverage all available NFL player movement data (both labeled and unlabeled) for better model performance.

## Overview

The NFL dataset contains:
- **Labeled data** (~46k sequences): `player_to_predict=True` with trajectory labels
- **Unlabeled data** (~200k+ sequences): `player_to_predict=False` without trajectory labels

Traditional supervised learning only uses the labeled data. This unsupervised pipeline allows you to:
1. **Pre-train** on ALL data (labeled + unlabeled) to learn good representations
2. **Fine-tune** on labeled data for the trajectory prediction task
3. **Improve performance** by leveraging 4-5x more training data

## Components

### 1. `unsupervised_data_loader.py`
- **UnsupervisedNFLDataLoader**: Loads both labeled and unlabeled sequences
- **UnsupervisedNFLSequence**: Keras data generator for unsupervised tasks

### 2. `unsupervised_models.py`
- **LSTMAutoencoder**: Autoencoder for representation learning
- **NextStepPredictor**: Self-supervised next-step prediction
- **transfer_encoder_weights**: Transfer pretrained weights to supervised models

### 3. `unsupervised_pretraining.py`
Main training script with command-line interface

## Usage

### Quick Start

#### 1. Train an Autoencoder (Recommended)

```bash
conda activate tensorflow

# Train on ALL data (labeled + unlabeled)
python src/manual_data_processing/unsupervised_pretraining.py \
    --task autoencoder \
    --epochs 50 \
    --batch_size 64 \
    --latent_dim 128
```

#### 2. Train Next-Step Predictor (Alternative)

```bash
python src/manual_data_processing/unsupervised_pretraining.py \
    --task next_step \
    --epochs 50 \
    --batch_size 64 \
    --prediction_steps 5
```

### Advanced Options

```bash
python src/manual_data_processing/unsupervised_pretraining.py \
    --task autoencoder \
    --epochs 100 \
    --batch_size 128 \
    --latent_dim 256 \
    --val_split 0.2 \
    --include_labeled \
    --include_unlabeled \
    --output_dir models/unsupervised
```

## Integration with Supervised Training

### Method 1: Load Pretrained Encoder

```python
from tensorflow import keras
from unsupervised_models import transfer_encoder_weights

# Load your supervised model
supervised_model = keras.models.load_model('path/to/supervised_model.keras')

# Load pretrained encoder
pretrained_encoder = keras.models.load_model('models/unsupervised/autoencoder_encoder.keras')

# Transfer weights (optionally freeze encoder layers)
supervised_model = transfer_encoder_weights(
    pretrained_encoder, 
    supervised_model,
    freeze_encoder=False  # Set to True to freeze pretrained weights
)

# Continue training
supervised_model.fit(train_data, epochs=50)
```

### Method 2: Initialize New Supervised Model

```python
from tensorflow.keras import layers, Model
from unsupervised_models import LSTMAutoencoder

# Build and pre-train autoencoder
ae = LSTMAutoencoder(input_shape=(28, 18), latent_dim=128)
ae.build_autoencoder()
# ... train autoencoder ...

# Build supervised model using pretrained encoder
inputs = layers.Input(shape=(28, 18))
features = ae.encoder(inputs)  # Use pretrained encoder
features = layers.Dense(256, activation='relu')(features)
features = layers.RepeatVector(10)(features)
features = layers.LSTM(128, return_sequences=True)(features)
outputs = layers.TimeDistributed(layers.Dense(2))(features)

supervised_model = Model(inputs, outputs)
supervised_model.compile(optimizer='adam', loss='mse')
```

## Training Strategies

### Strategy 1: Full Unsupervised Pre-training
1. Pre-train autoencoder on ALL data (labeled + unlabeled)
2. Transfer encoder weights to supervised model
3. Fine-tune on labeled data

**Best for**: Maximum utilization of unlabeled data

### Strategy 2: Semi-supervised Learning
1. Pre-train on unlabeled data only
2. Transfer weights
3. Train on labeled data

**Best for**: When labeled and unlabeled data distributions differ

### Strategy 3: Multi-task Learning
1. Combine supervised and unsupervised objectives
2. Train jointly on both tasks

**Best for**: End-to-end learning

## Expected Results

Based on typical transfer learning outcomes:

- **Baseline** (supervised only): Using ~46k labeled sequences
- **With pre-training**: Using ~250k total sequences
  - Expected improvement: 10-30% better validation metrics
  - Faster convergence (fewer epochs needed)
  - Better generalization (less overfitting)

## Model Architectures

### LSTM Autoencoder
```
Encoder:
  Input: (timesteps, features)
  → LSTM(256) → Dropout(0.2)
  → LSTM(128) → Dropout(0.2)
  → Dense(latent_dim)

Decoder:
  Input: (latent_dim,)
  → RepeatVector(timesteps)
  → LSTM(128) → Dropout(0.2)
  → LSTM(256) → Dropout(0.2)
  → TimeDistributed(Dense(features))
```

### Next-Step Predictor
```
Input: (timesteps, features)
→ LSTM(256) → Dropout(0.2)
→ LSTM(128) → Dropout(0.2)
→ RepeatVector(prediction_steps)
→ LSTM(128)
→ TimeDistributed(Dense(features))
Output: (prediction_steps, features)
```

## Monitoring Training

The training script provides:
- Real-time loss and metrics
- Model checkpointing (saves best model)
- Early stopping (stops if no improvement)
- Learning rate reduction on plateau

## Output Files

After training, you'll find:
```
models/unsupervised/
├── autoencoder_20250126_073000.keras      # Full autoencoder
├── autoencoder_20250126_073000_encoder.keras  # Encoder only (for transfer)
└── next_step_20250126_074500.keras       # Next-step predictor
```

## Troubleshooting

### OOM (Out of Memory) Errors
- Reduce `--batch_size` (try 32 or 16)
- Reduce `--maxlen` to limit sequence length
- Train on fewer sequences initially

### Poor Reconstruction Quality
- Increase `--latent_dim` (try 256)
- Add more LSTM units
- Train for more epochs
- Check for data quality issues

### Slow Training
- Increase `--batch_size` if memory allows
- Use GPU if available
- Reduce dataset size for testing

## Next Steps

1. **Start with autoencoder** on all data (recommended)
2. **Evaluate** reconstruction quality on validation set
3. **Transfer** encoder weights to supervised model
4. **Compare** supervised model with/without pre-training
5. **Iterate** on architecture and hyperparameters

## References

- LSTM Autoencoders: https://arxiv.org/abs/1502.04681
- Self-supervised Learning: https://arxiv.org/abs/2002.05709
- Transfer Learning: https://arxiv.org/abs/1411.1792
