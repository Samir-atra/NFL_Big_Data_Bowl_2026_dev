# NFL Big Data Bowl 2026 - Player Trajectory Prediction

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

A comprehensive machine learning solution for predicting NFL player trajectories using deep learning. This repository implements two distinct approaches: a **Manual LSTM-based approach** with unsupervised pre-training and a **Transformer-based approach** with scikit-learn preprocessing.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Two Approaches](#two-approaches)
  - [Approach 1: Manual LSTM with Unsupervised Pre-training](#approach-1-manual-lstm-with-unsupervised-pre-training)
  - [Approach 2: Transformer-based with Sklearn Preprocessing](#approach-2-transformer-based-with-sklearn-preprocessing)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Submission](#submission)
- [Documentation](#documentation)
- [Contributing](#contributing)

---

## 🎯 Overview

This project tackles the NFL Big Data Bowl 2026 challenge: predicting future player positions (x, y coordinates) based on historical tracking data. The solution leverages:

- **Deep Learning**: LSTM and Transformer architectures for sequence modeling
- **Transfer Learning**: Unsupervised pre-training on unlabeled data
- **Efficient Data Processing**: Polars and custom data loaders for large-scale CSV processing
- **Hyperparameter Optimization**: Keras Tuner with Hyperband algorithm
- **Production-Ready Code**: Modular design following Google's Python style guide

### Key Features

✅ **Two Complete Implementations**: Manual LSTM and Transformer approaches  
✅ **Unsupervised Pre-training**: Leverage 250k+ unlabeled sequences  
✅ **Hyperparameter Tuning**: Automated search with Keras Tuner  
✅ **Data Caching**: Optimized data loading with caching mechanisms  
✅ **Comprehensive Documentation**: Detailed notebooks and markdown guides  
✅ **Kaggle-Ready**: Submission notebooks for direct competition use  

---

## 📁 Project Structure

```
NFL_Big_Data_Bowl_2026_dev/
├── README.md                          # This file
├── SEQUENCE_LENGTH_EXPLANATION.md     # Detailed sequence length analysis
├── LICENSE                            # Apache 2.0 License
│
├── src/                               # Source code
│   ├── manual_data_processing/        # Approach 1: Manual LSTM implementation
│   │   ├── UNSUPERVISED_README.md     # Unsupervised pre-training guide
│   │   ├── unsupervised_pretraining.py # Pre-training script
│   │   ├── unsupervised_models.py     # Autoencoder & NextStep models
│   │   ├── unsupervised_data_loader.py # Data loader for unsupervised learning
│   │   ├── csv_to_numpy.py            # Polars-based CSV to NumPy converter
│   │   ├── csv_to_keras_sequence.py   # Keras Sequence data generator
│   │   ├── hyperband_unsupervised.py  # Hyperparameter tuning
│   │   ├── manual_training.ipynb      # Complete training notebook (documented)
│   │   ├── manual_submission.ipynb    # Kaggle submission notebook
│   │   └── build_best_encoder.py      # Encoder extraction utility
│   │
│   ├── data_loader.py                 # Approach 2: Sklearn-based data loader
│   ├── model.py                       # Transformer model with preprocessing
│   ├── predictor.py                   # Main training script (Approach 2)
│   ├── evaluate_model.py              # Model evaluation utilities
│   ├── keras_tuner_runner.py          # Hyperparameter tuning runner
│   │
│   ├── data_caching/                  # Data caching optimization
│   │   ├── README_DATA_CACHING.md     # Caching documentation
│   │   └── OPTIMIZATION_SUMMARY.md    # Performance optimization guide
│   │
│   ├── patches/                       # Kaggle-specific patches
│   │   └── KAGGLE_CACHE_GUIDE.md      # Kaggle caching guide
│   │
│   └── visualizations/                # Data visualization tools
│
├── nfl-big-data-bowl-2026-prediction/ # Prediction dataset
├── nfl-big-data-bowl-2026-analytics/  # Analytics dataset
│
├── trained_models/                    # Saved model checkpoints
├── submissions/                       # Competition submissions
├── tuner_results/                     # Hyperparameter tuning results
├── cached_data/                       # Cached preprocessed data
│
└── Docs/                              # Additional documentation
    └── notes.txt                      # Development notes
```

---

## 🔬 Two Approaches

This repository implements two distinct approaches to tackle the trajectory prediction problem. Each has unique strengths and use cases.

### Approach 1: Manual LSTM with Unsupervised Pre-training

**Location**: `src/manual_data_processing/`

#### Philosophy
Maximize data utilization by leveraging **all available player sequences** (labeled + unlabeled) through unsupervised pre-training, then fine-tune on labeled data.

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  STAGE 1: Unsupervised Pre-training         │
├─────────────────────────────────────────────────────────────┤
│  ALL Data (250k+ sequences)                                 │
│         ↓                                                    │
│  LSTM Autoencoder                                           │
│    Encoder: [LSTM(256) → LSTM(128) → Dense(latent)]        │
│    Decoder: [RepeatVector → LSTM(128) → LSTM(256)]         │
│         ↓                                                    │
│  Learn Representations (Reconstruction Task)                │
│         ↓                                                    │
│  Save Encoder Weights                                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  STAGE 2: Supervised Fine-tuning            │
├─────────────────────────────────────────────────────────────┤
│  Labeled Data (46k sequences)                               │
│         ↓                                                    │
│  Load Pre-trained Encoder                                   │
│         ↓                                                    │
│  Seq2Seq Model:                                             │
│    [Pre-trained Encoder → LSTM(256) → Dense(2)]            │
│         ↓                                                    │
│  Fine-tune on (x, y) Prediction                             │
│         ↓                                                    │
│  Final Model                                                │
└─────────────────────────────────────────────────────────────┘
```

#### Key Components

| Component | File | Description |
|-----------|------|-------------|
| **Data Loader** | `unsupervised_data_loader.py` | Polars-based loader for all sequences |
| **Models** | `unsupervised_models.py` | LSTMAutoencoder, NextStepPredictor |
| **Training** | `unsupervised_pretraining.py` | CLI for pre-training |
| **Fine-tuning** | `manual_training.ipynb` | Complete pipeline notebook |
| **Submission** | `manual_submission.ipynb` | Kaggle submission |

#### Advantages

- ✅ **4-5x More Training Data**: Uses unlabeled sequences
- ✅ **Better Generalization**: Pre-trained representations
- ✅ **Faster Convergence**: Warm-start from pre-training
- ✅ **Modular Design**: Separate pre-training and fine-tuning
- ✅ **Transfer Learning**: Encoder reusable for other tasks

#### Usage

```bash
# Step 1: Unsupervised Pre-training
conda activate tensorflow
python src/manual_data_processing/unsupervised_pretraining.py \
    --task autoencoder \
    --epochs 50 \
    --batch_size 64 \
    --latent_dim 128

# Step 2: Supervised Fine-tuning (in notebook)
jupyter notebook src/manual_data_processing/manual_training.ipynb
```

**See**: [`src/manual_data_processing/UNSUPERVISED_README.md`](src/manual_data_processing/UNSUPERVISED_README.md) for detailed documentation.

---

### Approach 2: Transformer-based with Sklearn Preprocessing

**Location**: `src/` (root level files)

#### Philosophy
End-to-end supervised learning with robust feature engineering using scikit-learn's preprocessing pipeline and efficient TensorFlow data loading.

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Processing Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│  Raw CSV Files                                              │
│         ↓                                                    │
│  Feature Engineering:                                       │
│    - Height → Inches                                        │
│    - Birth Date → Age                                       │
│    - Categorical → One-Hot Encoding                         │
│    - Numerical → StandardScaler                             │
│         ↓                                                    │
│  ColumnTransformer (sklearn)                                │
│         ↓                                                    │
│  Sequence Creation (sliding window)                         │
│         ↓                                                    │
│  TensorFlow Dataset                                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                      Model Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  Input: (batch, 10, features)                               │
│         ↓                                                    │
│  LSTM Layers (configurable depth)                           │
│         ↓                                                    │
│  Dense Layers                                               │
│         ↓                                                    │
│  Output: (batch, 2)  # (x, y) coordinates                  │
└─────────────────────────────────────────────────────────────┘
```

#### Key Components

| Component | File | Description |
|-----------|------|-------------|
| **Data Loader** | `data_loader.py` | Sklearn ColumnTransformer pipeline |
| **Model** | `model.py` | LSTM model with preprocessing |
| **Training** | `predictor.py` | Main training script |
| **Tuning** | `keras_tuner_runner.py` | Hyperparameter optimization |
| **Evaluation** | `evaluate_model.py` | Model evaluation utilities |

#### Advantages

- ✅ **Production-Ready**: Sklearn preprocessing is industry standard
- ✅ **Reproducible**: Fitted preprocessor saved with model
- ✅ **Efficient**: TensorFlow Dataset API for data loading
- ✅ **Flexible**: Easy to modify feature engineering
- ✅ **Hyperparameter Tuning**: Integrated Keras Tuner support

#### Usage

```bash
# Training
conda activate tensorflow
python src/predictor.py

# Hyperparameter Tuning
python src/keras_tuner_runner.py

# Evaluation
python src/evaluate_model.py
```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.12+
- **Conda**: For environment management
- **GPU**: Recommended (NVIDIA with CUDA support)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/NFL_Big_Data_Bowl_2026_dev.git
cd NFL_Big_Data_Bowl_2026_dev
```

2. **Create conda environment**
```bash
conda create -n tensorflow python=3.12
conda activate tensorflow
```

3. **Install dependencies**
```bash
pip install tensorflow==2.18.0
pip install polars pandas numpy scikit-learn
pip install keras-tuner matplotlib jupyter
pip install joblib
```

4. **Download data**
```bash
# Download from Kaggle:
# https://www.kaggle.com/competitions/nfl-big-data-bowl-2026

# Extract to:
# - nfl-big-data-bowl-2026-prediction/
# - nfl-big-data-bowl-2026-analytics/
```

---

## 🏃 Quick Start

### Approach 1: Manual LSTM (Recommended for Best Performance)

```bash
# 1. Pre-train on all data
python src/manual_data_processing/unsupervised_pretraining.py \
    --task autoencoder \
    --epochs 50

# 2. Fine-tune on labeled data
jupyter notebook src/manual_data_processing/manual_training.ipynb
# Run all cells in the notebook

# 3. Generate submission
jupyter notebook src/manual_data_processing/manual_submission.ipynb
```

### Approach 2: Transformer-based (Simpler, Faster Setup)

```bash
# 1. Train model
python src/predictor.py

# 2. Evaluate
python src/evaluate_model.py

# 3. Submit
jupyter notebook src/submission.ipynb
```

---

## 📊 Data

### Dataset Overview

| Dataset | Sequences | Players | Frames | Size |
|---------|-----------|---------|--------|------|
| **Prediction** | ~46k labeled | 22 per play | Variable | ~107 MB |
| **Analytics** | ~200k+ unlabeled | 22 per play | Variable | ~108 MB |
| **Total** | ~250k | - | - | ~215 MB |

### Data Structure

**Input Features** (18 total):
- Player attributes: height, weight, age, position
- Movement: x, y, speed, acceleration, direction
- Game context: down, distance, quarter, score
- Team: offense/defense, formation

**Output Labels**:
- `x`: Future x-coordinate
- `y`: Future y-coordinate

**Sequence Format**:
- Input: 10 timesteps of historical data
- Output: Single timestep prediction

See [`SEQUENCE_LENGTH_EXPLANATION.md`](SEQUENCE_LENGTH_EXPLANATION.md) for detailed analysis.

---

## 🎓 Training

### Approach 1: Two-Stage Training

#### Stage 1: Unsupervised Pre-training

```bash
python src/manual_data_processing/unsupervised_pretraining.py \
    --task autoencoder \
    --epochs 100 \
    --batch_size 64 \
    --latent_dim 128 \
    --include_labeled \
    --include_unlabeled
```

**Output**:
- `autoencoder_{timestamp}.keras` - Full model
- `autoencoder_{timestamp}_encoder.keras` - Encoder for transfer

#### Stage 2: Supervised Fine-tuning

Open `src/manual_data_processing/manual_training.ipynb` and run cells for:
1. Load pre-trained encoder
2. Build supervised model
3. Transfer weights
4. Fine-tune on labeled data

### Approach 2: Direct Supervised Training

```bash
python src/predictor.py
```

**Configuration** (edit in `predictor.py`):
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size
- `LEARNING_RATE`: Initial learning rate
- `SEQUENCE_LENGTH`: Input sequence length

### Hyperparameter Tuning

```bash
python src/keras_tuner_runner.py
```

**Tunable Parameters**:
- LSTM units: [64, 128, 256, 512]
- Number of layers: [2, 3, 4, 5, 6]
- Learning rate: [1e-4, 1e-2] (log scale)
- Dropout: [0.1, 0.5]

---

## 📈 Evaluation

### Metrics

- **Primary**: Mean Squared Error (MSE)
- **Secondary**: Mean Absolute Error (MAE)
- **Competition**: Custom Kaggle metric

### Evaluation Script

```bash
python src/evaluate_model.py
```

**Outputs**:
- Validation loss curves
- Prediction vs. ground truth plots
- Error distribution analysis

---

## 📤 Submission

### Generate Submission File

**Approach 1**:
```bash
jupyter notebook src/manual_data_processing/manual_submission.ipynb
```

**Approach 2**:
```bash
jupyter notebook src/submission.ipynb
```

**Output**: `submission.csv` ready for Kaggle upload

### Submission Format

```csv
row_id,x,y
0,45.23,12.67
1,46.12,13.45
...
```

---

## 📚 Documentation

### Main Documentation

- **[UNSUPERVISED_README.md](src/manual_data_processing/UNSUPERVISED_README.md)**: Complete guide to unsupervised pre-training
- **[SEQUENCE_LENGTH_EXPLANATION.md](SEQUENCE_LENGTH_EXPLANATION.md)**: Sequence length analysis
- **[DATA_CACHING.md](src/data_caching/README_DATA_CACHING.md)**: Data caching optimization
- **[KAGGLE_CACHE_GUIDE.md](src/patches/KAGGLE_CACHE_GUIDE.md)**: Kaggle-specific caching

### Notebooks

All notebooks include comprehensive markdown documentation:

- **`manual_training.ipynb`**: Fully documented training pipeline
- **`manual_submission.ipynb`**: Submission generation
- **`predictor.ipynb`**: Alternative training notebook

### Code Documentation

All code follows [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html):
- Module-level docstrings
- Function/class documentation
- Type hints
- Inline comments for complex logic

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow Google Python Style Guide
- Add docstrings to all functions/classes
- Include type hints
- Update documentation for new features
- Test on both CPU and GPU

---

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NFL Big Data Bowl 2026**: Kaggle competition organizers
- **TensorFlow Team**: For the excellent deep learning framework
- **Polars**: For high-performance data processing
- **Keras Tuner**: For hyperparameter optimization

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.

---

## 🔗 Useful Links

- [Competition Page](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Tuner Guide](https://keras.io/keras_tuner/)
- [Polars Documentation](https://pola-rs.github.io/polars/)

---

## Citation

If you use this project in your research, please cite it as follows:

 Attrah, Samer, Supervised LSTM Sequence-to-Sequence Models for NFL Player Trajectory Prediction: A Deep Learning Approach to the Big Data Bowl 2026 (April 20, 2026). Available at SSRN: https://ssrn.com/abstract=6749101 or http://dx.doi.org/10.2139/ssrn.6749101 

---

**Happy Coding! 🏈📊**
