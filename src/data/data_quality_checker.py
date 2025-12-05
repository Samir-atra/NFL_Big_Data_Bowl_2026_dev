
"""
Module for checking the quality of NFL Big Data Bowl datasets.

This module defines the DataQualityChecker class, which provides utilities
for generating reports on data quality issues such as missing values (NaN)
and empty/sparse values (including empty strings, whitespace, and pandas.NA).
It operates on a dictionary of pandas DataFrames, allowing for consistent
quality checking across multiple weekly datasets.
"""
import pandas as pd
import numpy as np
import os

class DataQualityChecker:
    """
    A class for checking data quality, including missing values (NaNs) and sparsity (zero values).
    """

    def __init__(self, data_frames):
        """
        Initializes the DataQualityChecker with a dictionary of DataFrames.

        Args:
            data_frames (dict): A dictionary where keys are identifiers (e.g., week numbers)
                                and values are pandas DataFrames.
        """
        self.data_frames = data_frames

    def get_missing_value_report(self):
        """
        Generates a report on missing values for each loaded DataFrame.

        Returns:
            dict: A dictionary where keys are week identifiers and values are DataFrames
                  showing missing value counts and percentages per column.
        """
        missing_reports = {}
        for week, df in self.data_frames.items():
            missing_count = df.isnull().sum()
            missing_percentage = (df.isnull().sum() / len(df)) * 100
            missing_info = pd.DataFrame({
                'Missing Count': missing_count,
                'Missing Percentage': missing_percentage
            })
            missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(
                by='Missing Count', ascending=False
            )
            missing_reports[week] = missing_info
        return missing_reports

    def get_empty_value_report(self):
        """
        Generates a report on empty values. This includes empty strings, whitespace-only strings
        in object columns, and pandas.NA values in any column type.
        Note: np.nan values are covered by get_missing_value_report.

        Returns:
            dict: A dictionary where keys are week identifiers and values are DataFrames
                  showing empty value counts and percentages per column.
        """
        empty_reports = {}
        print(f"Generating empty value report for object columns and pandas.NA.")
        for week, df in self.data_frames.items():
            empty_cols_info = []
            
            for col in df.columns:
                empty_count = 0
                total_count = len(df[col])

                if total_count == 0:
                    continue

                # Check for empty strings and whitespace-only strings in object columns
                if df[col].dtype == 'object':
                    empty_count += df[col].apply(lambda x: isinstance(x, str) and (x == '' or x.isspace())).sum()
                
                # Check for pandas.NA in any column type
                empty_count += df[col].isin([pd.NA]).sum()
                
                if empty_count > 0:
                    empty_percentage = (empty_count / total_count) * 100
                    empty_cols_info.append({
                        'Column': col,
                        'Empty Count': empty_count,
                        'Empty Percentage': empty_percentage
                    })
            
            if empty_cols_info:
                empty_info_df = pd.DataFrame(empty_cols_info).sort_values(
                    by='Empty Percentage', ascending=False
                )
                empty_reports[week] = empty_info_df
            else:
                empty_reports[week] = pd.DataFrame(columns=['Column', 'Empty Count', 'Empty Percentage'])
        
        return empty_reports

if __name__ == '__main__':
    # Example Usage: Create dummy dataframes for demonstration
    print("Running DataQualityChecker example...")
    dummy_data_1 = {
        'col_num_1': [1, 2, np.nan, 4, 0, 0, 0, 8, 9, 10],
        'col_num_2': [0, 0, pd.NA, 0, 5, 6, 7, 0, 0, 10], # Added pd.NA
        'col_cat_1': ['A', 'B', 'A', np.nan, 'C', ' ', 'B', '', 'A', 'B'], # Added empty and whitespace string, np.nan
        'col_cat_2': ['X', '', 'Y', 'Z', pd.NA, 'W', ' ', 'X', 'Y', 'Z'] # Added empty and whitespace string, pd.NA
    }
    dummy_data_2 = {
        'col_num_1': [11, 12, 13, 14, 0, 0, 0, 0, 0, 0],
        'col_num_3': [1, np.nan, 3, 4, pd.NA, 6, 7, 8, 9, 10] # Added pd.NA
    }
    
    df_week_1 = pd.DataFrame(dummy_data_1)
    df_week_2 = pd.DataFrame(dummy_data_2)
    
    test_data_frames = {
        'week_1': df_week_1,
        'week_2': df_week_2
    }

    checker = DataQualityChecker(test_data_frames)

    print("\n--- Missing Value Report ---")
    missing_report = checker.get_missing_value_report()
    for week, report in missing_report.items():
        if not report.empty:
            print(f"  {week}:\n{report}")
        else:
            print(f"  {week}: No missing values.")

    print("\n--- Empty Value Report ---")
    empty_report = checker.get_empty_value_report()
    for week, report in empty_report.items():
        if not report.empty:
            print(f"  {week}:\n{report}")
        else:
            print(f"  {week}: No empty values found.")
