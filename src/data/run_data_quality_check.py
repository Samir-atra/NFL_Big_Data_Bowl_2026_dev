"""
Script to run a comprehensive data quality check on the NFL Big Data Bowl datasets.

This module is a standalone script that loads raw input data for specified
weeks and uses the DataQualityChecker class to generate reports on missing
values (NaN) and empty values (empty strings, whitespace, pandas.NA).
It is intended for quick assessment of data integrity before processing or training.
"""
import pandas as pd
import numpy as np
import os
from data_quality_checker import DataQualityChecker

def run_quality_check(data_dir, week_numbers=range(1, 19)):
    data_frames = {}
    print(f"Loading data from: {data_dir}")
    for week in week_numbers:
        file_name = f"input_2023_w{week:02d}.csv"
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            print(f"  Loading {file_name}...")
            data_frames[f"week_{week}"] = pd.read_csv(file_path)
        else:
            print(f"  Warning: File not found: {file_name}")
    print("Data loading complete.")

    quality_checker = DataQualityChecker(data_frames)

    print("\n--- Missing Value Report ---")
    missing_report = quality_checker.get_missing_value_report()
    for week, report in missing_report.items():
        if not report.empty:
            print(f"  {week}:\n{report}")
        else:
            print(f"  {week}: No missing values.")

    print("\n--- Empty Value Report ---")
    empty_report = quality_checker.get_empty_value_report()
    for week, report in empty_report.items():
        if not report.empty:
            print(f"  {week}:\n{report}")
        else:
            print(f"  {week}: No empty values found.")

if __name__ == '__main__':
    data_directory = "nfl-big-data-bowl-2026-prediction/train"
    run_quality_check(data_directory)