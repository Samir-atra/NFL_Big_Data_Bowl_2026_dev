"""Training module for Lagrangian Neural Network on NFL prediction data.

This module coordinates data loading, sequence alignment, mini-batching, and
parameter optimization using JAX and Optax. It evaluates trajectory accuracy
using Mean Squared Error (MSE), Average Displacement Error (ADE), and Final
Displacement Error (FDE).
"""

import argparse
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

# Add repository root to Python path if necessary
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from src.LNN.data_loader import NFLDataLoader, pad_jax_sequences
    from src.LNN.lagrangian_network import FullLNNModel
except ModuleNotFoundError:
    from data_loader import NFLDataLoader, pad_jax_sequences
    from lagrangian_network import FullLNNModel



def compute_ade_fde(
    predictions: jax.Array,
    ground_truth: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Computes Average Displacement Error (ADE) and Final Displacement Error (FDE).

    Args:
        predictions: Predicted trajectories of shape (batch_size, horizon, 2).
        ground_truth: Ground truth trajectories of shape (batch_size, horizon, 2).

    Returns:
        Tuple of (mean_ade, mean_fde) scalar JAX arrays.
    """
    diff = predictions - ground_truth
    # Euclidean distances at each timestep: shape (batch_size, horizon)
    distances = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1) + 1e-8)
    ade = jnp.mean(distances)
    fde = jnp.mean(distances[:, -1])
    return ade, fde


def loss_fn(
    params: Dict[str, Any],
    model: FullLNNModel,
    batch_x: jax.Array,
    batch_y: jax.Array,
    horizon: int,
    x_idx: int = 0,
    y_idx: int = 1,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Calculates supervised trajectory prediction loss.

    Args:
        params: Model parameters.
        model: FullLNNModel instance.
        batch_x: Input tracking tensor of shape (batch_size, seq_len, in_dim).
        batch_y: Ground truth future trajectory tensor of shape (batch_size, horizon, 2).
        horizon: Prediction horizon length.
        x_idx: Index for x-coordinate feature in batch_x.
        y_idx: Index for y-coordinate feature in batch_x.

    Returns:
        Tuple of (scalar_loss, metrics_dictionary).
    """
    preds = model.predict_batch(batch_x, horizon, params, x_idx=x_idx, y_idx=y_idx)
    target = batch_y[:, :horizon, :]

    mse = jnp.mean(jnp.square(preds - target))
    ade, fde = compute_ade_fde(preds, target)

    # Regularization on large damping or energy divergence
    loss = mse
    metrics = {"loss": loss, "mse": mse, "ade": ade, "fde": fde}
    return loss, metrics


class LNNTrainer:
    """Trainer orchestrating LNN optimization and evaluation.

    Attributes:
        model: FullLNNModel instance.
        optimizer: Optax GradientTransformation optimizer.
        lr: Learning rate.
        horizon: Prediction horizon length.
        x_idx: Position x column index.
        y_idx: Position y column index.
    """

    def __init__(
        self,
        model: FullLNNModel,
        lr: float = 1e-3,
        horizon: int = 10,
        x_idx: int = 0,
        y_idx: int = 1,
    ):
        """Initializes LNNTrainer.

        Args:
            model: FullLNNModel instance.
            lr: Learning rate for Adam optimizer.
            horizon: Number of future timesteps to predict.
            x_idx: Column index of x-position in input features.
            y_idx: Column index of y-position in input features.
        """
        self.model = model
        self.lr = lr
        self.horizon = horizon
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.optimizer = optax.adam(learning_rate=lr)

        @jax.jit
        def step_fn(
            params: Dict[str, Any],
            opt_st: Any,
            bx: jax.Array,
            by: jax.Array,
        ) -> Tuple[Dict[str, Any], Any, jax.Array]:
            grad_fn = jax.value_and_grad(
                lambda p: loss_fn(
                    p, self.model, bx, by, self.horizon, self.x_idx, self.y_idx
                ),
                has_aux=True,
            )
            (b_loss, _), grads = grad_fn(params)
            updates, new_opt_st = self.optimizer.update(grads, opt_st, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_st, b_loss

        self._step_fn = step_fn

        @jax.jit
        def eval_step_fn(
            params: Dict[str, Any],
            bx: jax.Array,
            by: jax.Array,
        ) -> Dict[str, jax.Array]:
            _, metrics = loss_fn(
                params, self.model, bx, by, self.horizon, self.x_idx, self.y_idx
            )
            return metrics

        self._eval_step_fn = eval_step_fn

    def train_epoch(
        self,
        params: Dict[str, Any],
        opt_state: Any,
        batches_x: List[jax.Array],
        batches_y: List[jax.Array],
    ) -> Tuple[Dict[str, Any], Any, float]:
        """Runs one full training epoch over mini-batches.

        Args:
            params: Model parameters.
            opt_state: Optimizer state.
            batches_x: List of input mini-batch arrays.
            batches_y: List of target mini-batch arrays.

        Returns:
            Tuple of (updated_params, updated_opt_state, average_epoch_loss).
        """
        total_loss = 0.0
        n_batches = len(batches_x)
        if n_batches == 0:
            return params, opt_state, 0.0

        for bx, by in zip(batches_x, batches_y):
            params, opt_state, b_loss = self._step_fn(params, opt_state, bx, by)
            total_loss += float(b_loss)

        return params, opt_state, total_loss / n_batches

    def evaluate(
        self,
        params: Dict[str, Any],
        batches_x: List[jax.Array],
        batches_y: List[jax.Array],
    ) -> Dict[str, float]:
        """Evaluates model performance over a validation dataset.

        Args:
            params: Current model parameters.
            batches_x: Validation input batches.
            batches_y: Validation target batches.

        Returns:
            Dictionary containing averaged validation metrics (loss, mse, ade, fde).
        """
        if not batches_x:
            return {"loss": 0.0, "mse": 0.0, "ade": 0.0, "fde": 0.0}

        accum = {"loss": 0.0, "mse": 0.0, "ade": 0.0, "fde": 0.0}
        n_batches = len(batches_x)

        for bx, by in zip(batches_x, batches_y):
            metrics = self._eval_step_fn(params, bx, by)
            for k in accum:
                accum[k] += float(metrics[k])

        return {k: v / n_batches for k, v in accum.items()}



def create_mini_batches(
    x_list: List[jax.Array],
    y_list: List[jax.Array],
    batch_size: int,
    max_len_x: Optional[int] = 10,
    horizon: int = 10,
) -> Tuple[List[jax.Array], List[jax.Array]]:
    """Splits sequence lists into uniform padded mini-batches.

    Args:
        x_list: List of input sequences.
        y_list: List of output sequences.
        batch_size: Number of sequences per batch.
        max_len_x: Max time steps for input padding.
        horizon: Target trajectory length.

    Returns:
        Tuple of (batch_x_list, batch_y_list).
    """
    total = len(x_list)
    batches_x = []
    batches_y = []

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        cur_x = x_list[start_idx:end_idx]
        cur_y = y_list[start_idx:end_idx]

        padded_x = pad_jax_sequences(cur_x, max_len=max_len_x)
        padded_y = pad_jax_sequences(cur_y, max_len=horizon)

        batches_x.append(padded_x)
        batches_y.append(padded_y)

    return batches_x, batches_y


def train_lnn(
    train_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    context_dim: int = 16,
    horizon: int = 10,
    val_split: float = 0.2,
    normalize: bool = True,
    max_files: Optional[int] = None,
    save_path: Optional[str] = "lnn_model.pkl",
    seed: int = 42,
) -> Tuple[FullLNNModel, Dict[str, Any], Dict[str, Any]]:
    """Loads prediction data and trains a Lagrangian Neural Network model.

    Args:
        train_dir: Directory containing input and output CSV files.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        hidden_dim: Hidden dimension for LNN energy networks.
        context_dim: Dimension of latent historical context vector.
        horizon: Trajectory prediction horizon (timesteps).
        val_split: Fraction of sequences reserved for validation.
        normalize: Whether to apply Z-score normalization on features.
        max_files: Optional limit on CSV files to process.
        save_path: File path to persist best model checkpoint.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (model, best_params, training_history).
    """
    print(f"Loading NFL tracking data from: {train_dir}")
    loader = NFLDataLoader(train_dir=train_dir)
    x_list, y_list = loader.get_aligned_data(
        normalize=normalize, max_files=max_files, join_on_frame=False
    )

    n_samples = len(x_list)
    print(f"Total aligned sequences loaded: {n_samples}")
    if n_samples == 0:
        raise ValueError("No matching sequences could be extracted from the dataset.")

    # Determine x and y coordinate indices in feature list
    input_dim = x_list[0].shape[1]
    feature_cols = loader.feature_cols
    x_idx = feature_cols.index("x") if "x" in feature_cols else 0
    y_idx = feature_cols.index("y") if "y" in feature_cols else 1

    print(f"Input features per step: {input_dim}")
    print(f"Coordinates located at indices: x={x_idx}, y={y_idx}")

    # Train / Validation Split
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    x_train, y_train = x_list[:n_train], y_list[:n_train]
    x_val, y_val = x_list[n_train:], y_list[n_train:]

    print(f"Train samples: {len(x_train)}, Validation samples: {len(x_val)}")

    train_bx, train_by = create_mini_batches(
        x_train, y_train, batch_size=batch_size, horizon=horizon
    )
    val_bx, val_by = create_mini_batches(
        x_val, y_val, batch_size=batch_size, horizon=horizon
    )

    # Initialize model
    rng = jax.random.PRNGKey(seed)
    rng_init, _rng_train = jax.random.split(rng)

    model = FullLNNModel(
        input_dim=input_dim,
        coord_dim=2,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
    )
    params = model.init_params(rng_init)

    trainer = LNNTrainer(
        model=model,
        lr=lr,
        horizon=horizon,
        x_idx=x_idx,
        y_idx=y_idx,
    )
    opt_state = trainer.optimizer.init(params)

    best_val_loss = float("inf")
    best_params = params
    history = {"train_loss": [], "val_loss": [], "val_ade": [], "val_fde": []}

    print("\nStarting Lagrangian Neural Network Training...")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        params, opt_state, train_loss = trainer.train_epoch(
            params, opt_state, train_bx, train_by
        )
        val_metrics = trainer.evaluate(params, val_bx, val_by)
        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_ade"].append(val_metrics["ade"])
        history["val_fde"].append(val_metrics["fde"])

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val ADE: {val_metrics['ade']:.4f} | "
            f"Val FDE: {val_metrics['fde']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_params = params
            if save_path:
                checkpoint = {
                    "params": best_params,
                    "stats": loader.stats,
                    "feature_cols": loader.feature_cols,
                    "x_idx": x_idx,
                    "y_idx": y_idx,
                    "input_dim": input_dim,
                    "horizon": horizon,
                    "best_val_loss": best_val_loss,
                }
                with open(save_path, "wb") as f:
                    pickle.dump(checkpoint, f)

    print("=" * 65)
    print(f"Training Complete. Best Validation Loss: {best_val_loss:.4f}")
    if save_path:
        print(f"Model saved to: {save_path}")

    return model, best_params, history


def main():
    """Command-line interface entry point for training LNN."""
    parser = argparse.ArgumentParser(
        description="Train a Lagrangian Neural Network on NFL tracking data."
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Path to training CSV directory. Defaults to prediction_train_dir from .env.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="LNN hidden dimension.")
    parser.add_argument("--context-dim", type=int, default=16, help="Context dimension.")
    parser.add_argument("--horizon", type=int, default=10, help="Future prediction horizon.")
    parser.add_argument("--max-files", type=int, default=None, help="Max CSV files to read.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="src/LNN/lnn_model_checkpoint.pkl",
        help="Path to save trained model.",
    )
    args = parser.parse_args()

    train_dir = args.train_dir
    if not train_dir:
        # Check workspace .env file
        candidate = "nfl-big-data-bowl-2026-prediction/train"
        if os.path.isdir(candidate):
            train_dir = candidate
        else:
            env_path = ".env"
            if os.path.isfile(env_path):
                with open(env_path, "r") as f:
                    for line in f:
                        if line.startswith("prediction_train_dir="):
                            val = line.strip().split("=", 1)[1].strip("\"'")
                            if os.path.isdir(val):
                                train_dir = val
                                break

    if not train_dir or not os.path.isdir(train_dir):
        raise ValueError(
            f"Invalid training directory: '{train_dir}'. "
            "Please provide a valid --train-dir argument."
        )

    train_lnn(
        train_dir=train_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        horizon=args.horizon,
        max_files=args.max_files,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
