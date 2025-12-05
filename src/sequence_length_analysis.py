"""
Utility to analyze sequence lengths for supervised training data.

The script scans the training data directories, groups rows by the identifier
(`game_id`, `play_id`, `nfl_id`) to form player trajectories, and computes the
length of each trajectory. It then reports statistics and suggests an optimal
fixed sequence length that minimizes padding waste while keeping the majority of
sequences intact.

This analysis is crucial for determining an appropriate `SEQUENCE_LENGTH` constant,
which impacts model performance and efficiency by controlling how much padding
is introduced.

Usage:
-----
    python sequence_length_analysis.py <data_root> [--coverage COVERAGE]

Arguments:
    <data_root>:
        Path to the directory containing the CSV files used for supervised training
        (e.g., the `train` subdirectory of the prediction dataset).
    
    --coverage (float, optional):
        Desired proportion of sequences to keep without truncation. The script will
        suggest a sequence length such that this proportion of sequences fits entirely.
        Defaults to 0.95 (95%%).

Example:
--------
    python sequence_length_analysis.py ../nfl-big-data-bowl-2026-prediction/train --coverage 0.99

The script prints a concise report of sequence length statistics and the suggested
optimal length to standard output.
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

    This function walks through the specified directory, identifies all CSV files,
    reads them into pandas DataFrames, and concatenates them into a single DataFrame.
    A `_source` column is added to each DataFrame to track the original file path,
    which can be useful for debugging.

    Parameters:
    ----------
    root_dir : str
        The directory path to search for CSV files (including subdirectories).

    Returns:
    -------
    pd.DataFrame
        A single DataFrame containing the combined data from all CSV files found.
        Includes an additional `_source` column indicating the origin of each row.

    Raises:
    -------
    FileNotFoundError:
        If no CSV files are found within the specified `root_dir`.
    """
    csv_paths = []
    # Walk through the directory tree to find all CSV files.
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.csv'):
                csv_paths.append(os.path.join(dirpath, f))
    
    # Raise an error if no CSV files were found.
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {root_dir}")

    dfs = []
    # Read each CSV file and add a source column.
    for p in csv_paths:
        df = pd.read_csv(p, low_memory=False) # `low_memory=False` can prevent dtype warnings for large files
        df["_source"] = p # Add column to track the source file
        dfs.append(df)
    
    # Concatenate all DataFrames into a single one.
    return pd.concat(dfs, ignore_index=True)


def compute_sequence_lengths(df: pd.DataFrame) -> List[int]:
    """Compute the length of each player's trajectory sequence.

    A player's trajectory is defined by a unique combination of `game_id`,
    `play_id`, and `nfl_id`. This function groups the DataFrame by these
    identifiers and counts the number of rows (frames) within each group.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the game and player data.

    Returns:
    -------
    List[int]
        A list where each element is the length (number of frames) of a unique
        player sequence.

    Raises:
    -------
    KeyError:
        If any of the required grouping columns (`game_id`, `play_id`, `nfl_id`)
        are missing from the DataFrame.
    """
    # Define the columns that uniquely identify a player's sequence.
    required_cols = {"game_id", "play_id", "nfl_id"}
    # Check if all required columns are present in the DataFrame.
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for grouping: {missing}")
    
    # Group by game, play, and player ID, then count the number of entries in each group.
    # `.size()` returns a Series with lengths, which is then converted to a list.
    grouped = df.groupby(["game_id", "play_id", "nfl_id"])
    lengths = grouped.size()
    return lengths.tolist()


def suggest_optimal_length(lengths: List[int], coverage: float = 0.95) -> Tuple[int, int]:
    """Suggest a fixed sequence length that covers a desired proportion of sequences.

    This function sorts the sequence lengths and determines a length such that
    `coverage` proportion of sequences are less than or equal to this length.
    It also calculates how many sequences would exceed this suggested length.

    Parameters:
    ----------
    lengths : List[int]
        A list of all computed sequence lengths.
    coverage : float, optional
        The desired proportion (e.g., 0.95 for 95%%) of sequences that should fit
        within the suggested length without truncation. Defaults to 0.95.

    Returns:
    -------
    Tuple[int, int]
        A tuple containing:
        - The suggested fixed sequence length.
        - The count of sequences that are longer than the suggested length.

    Raises:
    -------
    ValueError:
        If the input `lengths` list is empty.
    """
    if not lengths:
        raise ValueError("Empty length list provided. Cannot suggest optimal length.")
    
    # Sort lengths in ascending order to easily find percentiles.
    sorted_lengths = sorted(lengths)
    
    # Calculate the index corresponding to the desired coverage (e.g., 95th percentile).
    # We subtract 1 because list indices are 0-based and coverage might point slightly past the last element.
    idx = int(len(sorted_lengths) * coverage) - 1
    # Ensure index is within bounds, especially for small datasets or coverage=1.0
    idx = max(0, min(idx, len(sorted_lengths) - 1))
    
    # The suggested length is the value at the calculated index.
    suggested = sorted_lengths[idx]
    
    # Count how many sequences are strictly longer than the suggested length.
    exceed = sum(1 for l in lengths if l > suggested)
    
    return suggested, exceed


def print_statistics(lengths: List[int], suggested: int, exceed: int) -> None:
    """Pretty-print sequence length statistics and the suggested optimal length.

    This function displays key statistics derived from the sequence lengths,
    including total count, average length, most common length, and detailed
    information about the suggested length and its coverage.

    Parameters:
    ----------
    lengths : List[int]
        The list of all computed sequence lengths.
    suggested : int
        The optimal sequence length suggested by `suggest_optimal_length`.
    exceed : int
        The count of sequences longer than the suggested length.
    """
    total = len(lengths)
    # Count the occurrences of each sequence length.
    counter = Counter(lengths)
    # Get the most common length and its count.
    most_common_len, most_common_cnt = counter.most_common(1)[0]
    # Calculate the average length.
    avg_len = sum(lengths) / total
    
    print("=== Sequence Length Statistics ===")
    print(f"Total sequences   : {total}")
    print(f"Average length    : {avg_len:.2f}")
    print(f"Most common length: {most_common_len} (appears {most_common_cnt} times)")
    # Report the suggested length and the proportion of data it covers.
    print(f"Suggested length  : {suggested} (covers ~{(total - exceed) / total:.2%} of data)")
    # Report the number and percentage of sequences that exceed the suggested length.
    print(f"Sequences longer than suggested: {exceed} ({exceed / total:.2%})")
    
    print("\nLength distribution (length: count) – top 10:")
    # Print the top 10 most frequent sequence lengths and their counts.
    for length, cnt in counter.most_common(10):
        print(f"  {length}: {cnt}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main function to parse arguments, load data, compute sequence lengths,
    and suggest an optimal fixed sequence length for model training.

    This script takes a directory path as input, finds all CSV files within it,
    computes the lengths of player trajectories, and then determines and reports
    an optimal sequence length based on a specified coverage percentage.
    """
    # --- Argument Parsing ---
    # Set up the argument parser to accept command-line arguments.
    parser = argparse.ArgumentParser(description="Analyse supervised training sequence lengths")
    
    # Add a positional argument for the root directory of the training data.
    parser.add_argument("data_root", help="Root directory containing training CSV files")
    
    # Add an optional argument for coverage percentage.
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.95,
        help="Fraction of sequences to keep without truncation (default: 0.95)",
    )
    
    # Parse the arguments provided by the user.
    args = parser.parse_args()

    # --- Data Processing and Analysis ---
    print(f"Loading CSV files from {args.data_root} …")
    # Load all CSV files into a single DataFrame.
    df = load_csv_files(args.data_root)
    
    print("Computing per-player sequence lengths …")
    # Compute the lengths of all player trajectories.
    lengths = compute_sequence_lengths(df)
    
    print(f"Suggesting optimal sequence length for {args.coverage:.0%} coverage ...")
    # Suggest an optimal sequence length based on the computed lengths and desired coverage.
    suggested, exceed = suggest_optimal_length(lengths, coverage=args.coverage)
    
    # Print the detailed statistics and the suggestion.
    print_statistics(lengths, suggested, exceed)


# This block ensures that `main()` is called only when the script is executed directly.
if __name__ == "__main__":
    main()
