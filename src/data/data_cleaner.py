
"""
Module for cleaning NFL Big Data Bowl datasets.

This module defines the DataCleaner class, which is responsible for loading
raw competition data and applying various imputation strategies (mean, median, mode,
forward/backward fill, constant) to handle missing values and data sparsity.
It also integrates with DataQualityChecker for pre- and post-cleaning data quality reports.
"""
import pandas as pd
import numpy as np
import os
from src.data_quality_checker import DataQualityChecker


class DataCleaner:
    """
    A class for cleaning NFL Big Data Bowl datasets, handling missing values and sparsity.
    """

    def __init__(self, data_dir):
        """
        Initializes the DataCleaner with the directory containing the raw data files.

        Args:
            data_dir (str): Path to the directory containing input CSV files.
        """
        self.data_dir = data_dir
        self.data_frames = {}

    def load_data(self, week_numbers=range(1, 19)):
        """
        Loads input data for specified week numbers into a dictionary of DataFrames.

        Args:
            week_numbers (iterable): An iterable of week numbers to load (e.g., range(1, 19)).
        """
        print(f"Loading data from: {self.data_dir}")
        for week in week_numbers:
            file_name = f"input_2023_w{week:02d}.csv"
            file_path = os.path.join(self.data_dir, file_name)
            if os.path.exists(file_path):
                print(f"  Loading {file_name}...")
                self.data_frames[f"week_{week}"] = pd.read_csv(file_path)
            else:
                print(f"  Warning: File not found: {file_name}")
        print("Data loading complete.")

    def fill_missing_values(self, strategy='median', numerical_cols=None, categorical_cols=None):
        """
        Fills missing values in the loaded DataFrames using specified strategies.

        Args:
            strategy (str): The imputation strategy ('mean', 'median', 'mode', 'ffill', 'bfill', 'constant').
                            For 'constant', fills with 0 for numerical and 'Missing' for categorical.
            numerical_cols (list, optional): List of numerical columns to apply strategy to. If None,
                                             it attempts to infer numerical columns.
            categorical_cols (list, optional): List of categorical columns to apply strategy to. If None,
                                                it attempts to infer categorical columns.
        """
        print(f"Filling missing values with strategy: '{strategy}'")
        for week, df in self.data_frames.items():
            print(f"  Processing week: {week}")

            # Infer column types if not provided
            if numerical_cols is None and categorical_cols is None:
                inferred_numerical = df.select_dtypes(include=np.number).columns
                inferred_categorical = df.select_dtypes(include='object').columns
            else:
                inferred_numerical = numerical_cols if numerical_cols is not None else []
                inferred_categorical = categorical_cols if categorical_cols is not None else []

            # Handle numerical columns
            for col in inferred_numerical:
                if df[col].isnull().any():
                    if strategy == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif strategy == 'median':                        df[col].fillna(df[col].median(), inplace=True)
                    elif strategy == 'ffill':
                        df[col].fillna(method='ffill', inplace=True)
                    elif strategy == 'bfill':
                        df[col].fillna(method='bfill', inplace=True)
                    elif strategy == 'constant':
                        df[col].fillna(0, inplace=True)
                    else:
                        print(f"    Warning: Unknown numerical strategy '{strategy}' for column '{col}'. Skipping.")

            # Handle categorical columns
            for col in inferred_categorical:
                if df[col].isnull().any():
                    if strategy == 'mode':
                        # Mode can return multiple values, pick the first one
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Missing'
                        df[col].fillna(mode_val, inplace=True)
                    elif strategy == 'constant':
                        df[col].fillna('Missing', inplace=True)
                    else:
                        print(f"    Warning: Unknown categorical strategy '{strategy}' for column '{col}'. Skipping.")

            self.data_frames[week] = df
        print("Missing value filling complete.")

    def get_cleaned_data(self):
        """
        Returns the dictionary of cleaned DataFrames.

        Returns:
            dict: A dictionary of cleaned pandas DataFrames.
        """
        return self.data_frames

if __name__ == '__main__':
    # Example Usage (assuming this script is run from the project root)
    data_directory = "nfl-big-data-bowl-2026-prediction/train"
    cleaner = DataCleaner(data_directory)
    cleaner.load_data(week_numbers=[1, 2]) # Load data for week 1 and 2 for demonstration

    # Initialize DataQualityChecker with the loaded dataframes
    quality_checker = DataQualityChecker(cleaner.get_cleaned_data())

    print("\n--- Initial Data Quality Report ---")
    print("\nMissing value report before cleaning:")
    missing_before = quality_checker.get_missing_value_report()
    for week, report in missing_before.items():
        if not report.empty:
            print(f"  {week}:\n{report}")
        else:
            print(f"  {week}: No missing values.")

    print("\nSparsity report before cleaning (50% threshold):")
    sparsity_before = quality_checker.get_sparsity_report(sparsity_threshold=50.0)
    for week, report in sparsity_before.items():
        if not report.empty:
            print(f"  {week}:\n{report}")
        else:
            print(f"  {week}: No sparse columns found above the threshold.")

    cleaner.fill_missing_values(strategy='median') # Fill with median for numerical, mode for categorical

    # Re-initialize DataQualityChecker with cleaned data to get updated reports
    quality_checker = DataQualityChecker(cleaner.get_cleaned_data())

    print("\n--- Data Quality Report After Cleaning ---")
    print("\nMissing value report after cleaning:")
    missing_after = quality_checker.get_missing_value_report()
    for week, report in missing_after.items():
        if not report.empty:
            print(f"  {week}:\n{report}")
        else:
            print(f"  {week}: No missing values.")
            
    print("\nSparsity report after cleaning (50% threshold) ---")
    sparsity_after = quality_checker.get_sparsity_report(sparsity_threshold=50.0)
    for week, report in sparsity_after.items():
        if not report.empty:
            print(f"  {week}:\n{report}")
        else:
            print(f"  {week}: No sparse columns found above the threshold.")

    # Access cleaned data
    cleaned_dfs = cleaner.get_cleaned_data()
    # print("\nCleaned DataFrame for week 1 (head):")
    # if "week_1" in cleaned_dfs:
    #     print(cleaned_dfs["week_1"].head())
