#!/usr/bin/env python3
"""Utility to analyse sequence lengths for supervised training data.

The script scans the training data directories, groups rows by the identifier
(`game_id`, `play_id`, `nfl_id`) to form player trajectories and computes the
length of each trajectory.  It then reports statistics and suggests an optimal
fixed sequence length that minimises padding waste while keeping the majority of
sequences intact.

Usage
-----
    python sequence_length_analysis.py <data_root>

where ``<data_root>`` points to the directory containing the CSV files used for
supervised training (e.g. ``.../train``).  The script prints a concise report
to stdout.
"""

import argparse
import os
from collections import Counter
from typing import List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_csv_files(root_dir: str) -> pd.DataFrame:
    """Load all CSV files under ``root_dir`` into a single DataFrame.

    Parameters
    ----------
    root_dir: str
        Directory containing CSV files (recursively searched).

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with an additional column ``_source`` indicating
        the originating file – useful for debugging.
    """
    csv_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.csv'):
                csv_paths.append(os.path.join(dirpath, f))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {root_dir}")

    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p, low_memory=False)
        df["_source"] = p
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def compute_sequence_lengths(df: pd.DataFrame) -> List[int]:
    """Return a list of trajectory lengths.

    The function groups by ``game_id``, ``play_id`` and ``nfl_id`` – the three
    columns that uniquely identify a player's sequence in the competition data.
    """
    required_cols = {"game_id", "play_id", "nfl_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for grouping: {missing}")
    grouped = df.groupby(["game_id", "play_id", "nfl_id"]).size()
    return grouped.tolist()


def suggest_optimal_length(lengths: List[int], coverage: float = 0.95) -> Tuple[int, int]:
    """Suggest a fixed sequence length.

    Parameters
    ----------
    lengths: List[int]
        List of raw sequence lengths.
    coverage: float, default 0.95
        Desired proportion of sequences that should fit without truncation.

    Returns
    -------
    Tuple[int, int]
        (suggested_length, number_of_sequences_exceeding_it)
    """
    if not lengths:
        raise ValueError("Empty length list")
    sorted_lengths = sorted(lengths)
    idx = int(len(sorted_lengths) * coverage) - 1
    suggested = sorted_lengths[idx]
    exceed = sum(1 for l in lengths if l > suggested)
    return suggested, exceed


def print_statistics(lengths: List[int], suggested: int, exceed: int) -> None:
    """Pretty‑print length statistics.
    """
    total = len(lengths)
    counter = Counter(lengths)
    most_common_len, most_common_cnt = counter.most_common(1)[0]
    avg_len = sum(lengths) / total
    print("=== Sequence Length Statistics ===")
    print(f"Total sequences   : {total}")
    print(f"Average length    : {avg_len:.2f}")
    print(f"Most common length: {most_common_len} (appears {most_common_cnt} times)")
    print(f"Suggested length  : {suggested} (covers ~{(total - exceed) / total:.2%} of data)")
    print(f"Sequences longer than suggested: {exceed} ({exceed / total:.2%})")
    print("\nLength distribution (length: count) – top 10:")
    for length, cnt in counter.most_common(10):
        print(f"  {length}: {cnt}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse supervised training sequence lengths")
    parser.add_argument("data_root", help="Root directory containing training CSV files")
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.95,
        help="Fraction of sequences to keep without truncation (default: 0.95)",
    )
    args = parser.parse_args()

    print(f"Loading CSV files from {args.data_root} …")
    df = load_csv_files(args.data_root)
    print("Computing per‑player sequence lengths …")
    lengths = compute_sequence_lengths(df)
    suggested, exceed = suggest_optimal_length(lengths, coverage=args.coverage)
    print_statistics(lengths, suggested, exceed)


if __name__ == "__main__":
    main()
