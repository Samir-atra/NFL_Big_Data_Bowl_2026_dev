"""Data loading and preprocessing pipeline for Lagrangian Neural Networks.

This module provides data loading, feature engineering, and sequence alignment
utilities for NFL Big Data Bowl 2026 player trajectory prediction.
All tabular transformations are performed using the Polars library, and all
numerical outputs are converted into JAX arrays.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import polars as pl


class NFLDataLoader:
    """Loads and preprocesses NFL Big Data Bowl 2026 tracking data using Polars.

    Filters input sequences for target players (`player_to_predict == True`),
    applies feature processing to categorical and continuous attributes, and
    aligns historical observations with future trajectory targets.

    Attributes:
        train_dir: Directory containing input and output CSV files.
        input_df: Polars DataFrame storing processed input rows.
        output_df: Polars DataFrame storing target trajectory rows.
        stats: Dictionary containing normalization mean and standard deviation.
        feature_cols: List of processed input feature column names.
    """

    def __init__(self, train_dir: str):
        """Initializes the NFLDataLoader with the directory of CSV files.

        Args:
            train_dir: Path to directory containing `input*.csv` and `output*.csv`.
        """
        self.train_dir = train_dir
        self.input_df: Optional[pl.DataFrame] = None
        self.output_df: Optional[pl.DataFrame] = None
        self.stats: Dict[str, Dict[str, float]] = {}
        self.feature_cols: List[str] = []

    def process_features(self, df: pl.DataFrame) -> Tuple[List[pl.Expr], List[str]]:
        """Constructs Polars expressions for feature engineering and type casting.

        Converts strings and booleans to floats, computes player age from birth date,
        and hashes categorical strings deterministically.

        Args:
            df: Raw Polars DataFrame containing tracking columns.

        Returns:
            A tuple of (expressions, feature_column_names) where expressions are
            Polars expressions to apply and feature_column_names are the names of
            the features being retained.
        """
        id_cols = {
            "game_id",
            "play_id",
            "nfl_id",
            "frame_id",
            "player_to_predict",
            "time",
            "player_name",
            "num_frames_output",
            "ball_land_x",
            "ball_land_y",
        }
        feature_cols = [col for col in df.columns if col not in id_cols]
        expressions: List[pl.Expr] = []

        for col in feature_cols:
            if col == "player_birth_date":
                # Convert birth date to approximate age relative to 2023 season
                age_expr = (
                    pl.lit("2023-09-01").str.to_date()
                    - pl.col(col).str.to_date(format="%Y-%m-%d", strict=False)
                ).dt.total_days() / 365.25
                expressions.append(age_expr.fill_null(25.0).cast(pl.Float64).alias(col))
                continue

            if df[col].dtype == pl.Utf8:
                expr = (
                    pl.when(pl.col(col).str.to_lowercase() == "true")
                    .then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "false")
                    .then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "left")
                    .then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "right")
                    .then(1.0)
                    .when(pl.col(col).str.to_lowercase() == "defense")
                    .then(0.0)
                    .when(pl.col(col).str.to_lowercase() == "offense")
                    .then(1.0)
                    .when(pl.col(col).str.contains(r"^\d+-\d+$"))
                    .then(
                        (
                            pl.col(col).str.extract(r"(\d+)-(\d+)", 1).cast(pl.Float64)
                            + pl.col(col).str.extract(r"(\d+)-(\d+)", 2).cast(pl.Float64)
                        )
                        / 2.0
                    )
                    .otherwise(
                        pl.col(col)
                        .cast(pl.Float64, strict=False)
                        .fill_null(
                            (pl.col(col).hash(seed=42) % 10000).cast(pl.Float64)
                        )
                    )
                    .cast(pl.Float64)
                    .alias(col)
                )
                expressions.append(expr)
            elif df[col].dtype == pl.Boolean:
                expressions.append(
                    pl.when(pl.col(col))
                    .then(1.0)
                    .otherwise(0.0)
                    .cast(pl.Float64)
                    .alias(col)
                )
            else:
                expressions.append(pl.col(col).cast(pl.Float64).alias(col))

        return expressions, feature_cols

    def load_input_files(self, max_files: Optional[int] = None) -> None:
        """Loads and filters input tracking CSV files.

        Reads `input*.csv` files, filters for rows where `player_to_predict` is true,
        and applies vectorized feature preprocessing.

        Args:
            max_files: Optional limit on the number of input files to load.
        """
        if not os.path.isdir(self.train_dir):
            raise FileNotFoundError(f"Directory not found: {self.train_dir}")

        input_files = sorted(
            [f for f in os.listdir(self.train_dir) if f.startswith("input") and f.endswith(".csv")]
        )
        if max_files is not None:
            input_files = input_files[:max_files]

        dfs: List[pl.DataFrame] = []
        for file_name in input_files:
            file_path = os.path.join(self.train_dir, file_name)
            try:
                df = pl.read_csv(file_path, infer_schema_length=10000)
                if "player_to_predict" in df.columns:
                    if df["player_to_predict"].dtype == pl.Boolean:
                        df = df.filter(pl.col("player_to_predict") == True)
                    else:
                        df = df.filter(
                            pl.col("player_to_predict").cast(pl.Utf8).str.to_lowercase() == "true"
                        )
                if df.height > 0:
                    dfs.append(df)
            except Exception as exc:
                print(f"Warning: Failed to load {file_name}: {exc}")

        if not dfs:
            self.input_df = pl.DataFrame()
            return

        combined_df = pl.concat(dfs, how="vertical_relaxed")
        feature_exprs, feature_cols = self.process_features(combined_df)
        self.feature_cols = feature_cols

        id_cols = ["game_id", "play_id", "nfl_id", "frame_id"]
        existing_id_cols = [c for c in id_cols if c in combined_df.columns]
        self.input_df = combined_df.select(
            [pl.col(c) for c in existing_id_cols] + feature_exprs
        )

    def load_output_files(self, max_files: Optional[int] = None) -> None:
        """Loads output tracking target CSV files.

        Extracts future position targets (columns 'x' and 'y') alongside sequence identifiers.

        Args:
            max_files: Optional limit on the number of output files to load.
        """
        if not os.path.isdir(self.train_dir):
            raise FileNotFoundError(f"Directory not found: {self.train_dir}")

        output_files = sorted(
            [f for f in os.listdir(self.train_dir) if f.startswith("output") and f.endswith(".csv")]
        )
        if max_files is not None:
            output_files = output_files[:max_files]

        dfs: List[pl.DataFrame] = []
        target_features = ["x", "y"]

        for file_name in output_files:
            file_path = os.path.join(self.train_dir, file_name)
            try:
                cols_to_read = ["game_id", "play_id", "nfl_id", "frame_id"] + target_features
                df = pl.read_csv(file_path, infer_schema_length=10000)
                existing_cols = [c for c in cols_to_read if c in df.columns]
                if all(c in existing_cols for c in ["game_id", "play_id", "nfl_id"] + target_features):
                    dfs.append(df.select(existing_cols))
            except Exception as exc:
                print(f"Warning: Failed to load {file_name}: {exc}")

        if not dfs:
            self.output_df = pl.DataFrame()
            return

        combined_df = pl.concat(dfs, how="vertical_relaxed")
        self.output_df = combined_df.with_columns(
            [pl.col(c).cast(pl.Float64) for c in target_features]
        )

    def normalize_features(
        self, df: pl.DataFrame, cols: List[str]
    ) -> Tuple[pl.DataFrame, Dict[str, Dict[str, float]]]:
        """Calculates Z-score normalization statistics and standardizes columns.

        Args:
            df: Polars DataFrame to normalize.
            cols: List of column names to standardize.

        Returns:
            A tuple of (normalized_df, stats_dict).
        """
        stats: Dict[str, Dict[str, float]] = {}
        exprs: List[pl.Expr] = []

        for col in cols:
            mean_val = df.select(pl.col(col).mean()).item()
            std_val = df.select(pl.col(col).std()).item()

            mean = 0.0 if mean_val is None else float(mean_val)
            std = 1.0 if std_val is None or std_val == 0.0 else float(std_val)

            stats[col] = {"mean": mean, "std": std}
            exprs.append(((pl.col(col) - mean) / std).alias(col))

        return df.with_columns(exprs), stats

    def get_aligned_data(
        self,
        normalize: bool = False,
        max_files: Optional[int] = None,
        join_on_frame: bool = False,
    ) -> Tuple[List[jax.Array], List[jax.Array]]:
        """Aligns input and output sequences, returning lists of JAX arrays.

        Args:
            normalize: Whether to apply Z-score normalization on input features.
            max_files: Limit on the number of CSV files to read.
            join_on_frame: If True, matches rows where frame_id is equal (frame-by-frame).
                If False, groups input by sequence keys to form history and aligns
                with the future trajectory sequence for that play.

        Returns:
            Tuple of (X_list, y_list) where each element is a JAX array representing
            one sequence: X of shape (seq_len_in, n_features) and y of shape
            (seq_len_out, 2) corresponding to (x, y) coordinates.
        """
        self.load_input_files(max_files=max_files)
        self.load_output_files(max_files=max_files)

        if self.input_df is None or self.input_df.is_empty():
            return [], []
        if self.output_df is None or self.output_df.is_empty():
            return [], []

        id_keys = ["game_id", "play_id", "nfl_id"]

        if join_on_frame:
            # Frame-by-frame alignment matching manual_training.ipynb inner join
            output_renamed = self.output_df.rename({"x": "target_x", "y": "target_y"})
            joined_df = self.input_df.join(
                output_renamed,
                on=id_keys + ["frame_id"],
                how="inner",
            ).sort(id_keys + ["frame_id"])

            feature_cols = [
                c
                for c in self.input_df.columns
                if c not in set(id_keys + ["frame_id", "target_x", "target_y"])
            ]
            target_cols = ["target_x", "target_y"]

            if normalize:
                joined_df, self.stats = self.normalize_features(joined_df, feature_cols)

            grouped = joined_df.group_by(id_keys, maintain_order=True).agg(
                [pl.col(c) for c in feature_cols + target_cols]
            )

            X_list: List[jax.Array] = []
            y_list: List[jax.Array] = []

            for row in grouped.iter_rows(named=True):
                seq_len = len(row[feature_cols[0]])
                if seq_len == 0:
                    continue

                feat_matrix = [row[c] for c in feature_cols]
                targ_matrix = [row[c] for c in target_cols]

                x_arr = jnp.array(feat_matrix, dtype=jnp.float32).T
                y_arr = jnp.array(targ_matrix, dtype=jnp.float32).T

                X_list.append(x_arr)
                y_list.append(y_arr)

            return X_list, y_list

        # Sequence-level alignment: input sequence -> future output trajectory
        input_feature_cols = [
            c for c in self.input_df.columns if c not in set(id_keys + ["frame_id"])
        ]
        target_cols = ["x", "y"]

        working_input = self.input_df
        if normalize:
            working_input, self.stats = self.normalize_features(working_input, input_feature_cols)

        sorted_input = working_input.sort(id_keys + ["frame_id"])
        grouped_input = sorted_input.group_by(id_keys, maintain_order=True).agg(
            [pl.col(c) for c in input_feature_cols]
        )

        sorted_output = (
            self.output_df.sort(id_keys + ["frame_id"])
            if "frame_id" in self.output_df.columns
            else self.output_df
        )
        grouped_output = sorted_output.group_by(id_keys, maintain_order=True).agg(
            [pl.col(c) for c in target_cols]
        )

        aligned_df = grouped_input.join(grouped_output, on=id_keys, how="inner")

        X_list = []
        y_list = []

        for row in aligned_df.iter_rows(named=True):
            in_len = len(row[input_feature_cols[0]])
            out_len = len(row[target_cols[0]])
            if in_len == 0 or out_len == 0:
                continue

            x_mat = [row[c] for c in input_feature_cols]
            y_mat = [row[c] for c in target_cols]

            x_arr = jnp.array(x_mat, dtype=jnp.float32).T
            y_arr = jnp.array(y_mat, dtype=jnp.float32).T

            X_list.append(x_arr)
            y_list.append(y_arr)

        return X_list, y_list


def pad_jax_sequences(
    sequences: List[jax.Array],
    max_len: Optional[int] = None,
    padding_value: float = 0.0,
) -> jax.Array:
    """Pads a list of 2D JAX arrays to uniform length along the time dimension.

    Args:
        sequences: List of arrays of shape (seq_len, num_features).
        max_len: Target sequence length. If None, max length of input is used.
        padding_value: Value used to pad shorter sequences.

    Returns:
        Padded JAX array of shape (num_sequences, max_len, num_features).
    """
    if not sequences:
        return jnp.empty((0, 0, 0), dtype=jnp.float32)

    actual_max = max(seq.shape[0] for seq in sequences)
    target_len = actual_max if max_len is None else max_len
    num_features = sequences[0].shape[1]
    num_samples = len(sequences)

    padded_list = []
    for seq in sequences:
        current_len = seq.shape[0]
        if current_len >= target_len:
            padded_list.append(seq[:target_len, :])
        else:
            pad_size = target_len - current_len
            padding = jnp.full((pad_size, num_features), padding_value, dtype=jnp.float32)
            padded_list.append(jnp.concatenate([seq, padding], axis=0))

    return jnp.stack(padded_list, axis=0)
