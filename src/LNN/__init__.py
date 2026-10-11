"""Lagrangian Neural Network (LNN) package for NFL player trajectory prediction.

This package provides data loading with Polars and physics-informed trajectory
modeling with JAX using the principle of stationary action and Euler-Lagrange
equations of motion.
"""

from src.LNN.data_loader import NFLDataLoader, pad_jax_sequences
from src.LNN.lagrangian_network import (
    ContextEncoder,
    FullLNNModel,
    LagrangianNeuralNetwork,
)
from src.LNN.train_lnn import LNNTrainer, train_lnn

__all__ = [
    "NFLDataLoader",
    "pad_jax_sequences",
    "ContextEncoder",
    "LagrangianNeuralNetwork",
    "FullLNNModel",
    "LNNTrainer",
    "train_lnn",
]
