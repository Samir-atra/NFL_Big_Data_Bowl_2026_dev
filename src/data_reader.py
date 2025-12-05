"""
This module provides classes and functions for reading and processing the
NFL Big Data Bowl 2026 dataset. It includes functionality to:
- Read data from specified directories.
- Identify available weeks within the dataset.
- Load input and output CSV files for specific weeks.
- Merge input and output data for a comprehensive view of each week.
"""

import pandas as pd
import os
import glob

class DataReader:
    """
    A class designed to read and manage the NFL Big Data Bowl 2026 dataset files.

    It provides methods to access weekly game data, distinguishing between
    input (play-by-play) and output (game results) information.
    """
    def __init__(self, data_dir: str):
        """
        Initializes the DataReader with the path to the root directory
        containing the dataset files.

        Args:
            data_dir (str): The absolute path to the directory where the dataset
                            (specifically the 'train' subdirectories containing CSVs)
                            is located.
        """
        self.data_dir = data_dir

    def get_weeks(self) -> list[str]:
        """
        Identifies and returns a sorted list of all available weeks present
        in the data directory based on the presence of 'input_*.csv' files.

        Returns:
            list[str]: A sorted list of week identifiers (e.g., ['01', '02', ...]).
        """
        weeks = set()
        # Use glob to find all csv files directly within the specified data directory.
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        
        for f_path in csv_files:
            filename = os.path.basename(f_path)
            # We identify a week by looking for files that start with 'input_'
            # and then extract the week identifier from the filename.
            if filename.startswith('input_'):
                # Extract the week number by removing the 'input_' prefix and '.csv' suffix.
                week_id = filename.replace('input_', '').replace('.csv', '')
                weeks.add(week_id)
        
        # Return the unique week identifiers, sorted chronologically.
        return sorted(list(weeks))

    def read_input(self, week: str) -> pd.DataFrame:
        """
        Reads the input data (play-by-play details) for a specific week
        into a pandas DataFrame.

        Args:
            week (str): The week identifier (e.g., '01', '02').

        Returns:
            pd.DataFrame: A DataFrame containing the input data for the specified week.
        """
        path = os.path.join(self.data_dir, f'input_{week}.csv')
        return pd.read_csv(path)

    def read_output(self, week: str) -> pd.DataFrame | None:
        """
        Reads the output data (game results or predictions) for a specific week
        into a pandas DataFrame, if the file exists.

        Args:
            week (str): The week identifier (e.g., '01', '02').

        Returns:
            pd.DataFrame or None: A DataFrame containing the output data for the
                                  specified week, or None if the output file
                                  does not exist for that week.
        """
        path = os.path.join(self.data_dir, f"output_{week}.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    def read_week(self, week: str) -> pd.DataFrame:
        """
        Reads both the input and output data for a given week and merges them
        into a single pandas DataFrame.

        Args:
            week (str): The week identifier (e.g., '01', '02').

        Returns:
            pd.DataFrame: A DataFrame containing the combined input and output
                          data for the specified week.
        """
        input_df = self.read_input(week)
        output_df = self.read_output(week) 
        
        if output_df is not None:
            # Concatenating input and output DataFrames.
            # This approach is robust because input and output files typically
            # cover distinct sets of frames: input usually contains frame 0,
            # while output contains frames > 0. pd.concat handles this efficiently.
            return pd.concat([input_df, output_df], ignore_index=True)
        # If no output data is found for the week, return only the input data.
        return input_df

def main():
    """
    Main function to demonstrate the functionality of the DataReader class.

    This function initializes DataReader instances for both the prediction
    and analytics datasets, lists the available weeks in each, and attempts
    to read and display the first few rows of data for the earliest week.
    It includes basic error handling for cases where data directories might
    not be accessible or contain expected files.
    """
    # Define the directory paths for the prediction and analytics datasets.
    # These paths are specific to the current project structure.
    prediction_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    analytics_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/'

    print("--- Reading Prediction Data ---")
    try:
        # Initialize DataReader for prediction data.
        prediction_reader = DataReader(prediction_data_dir)
        # Get list of weeks available in the prediction data.
        prediction_weeks = prediction_reader.get_weeks()
        print(f"Available weeks in prediction data: {prediction_weeks}")

        # Proceed if weeks are found.
        if prediction_weeks:
            # Select the first week found for demonstration.
            first_week = prediction_weeks[0]
            print(f"Reading data for week {first_week} from prediction data...")
            # Read the combined data for the selected week.
            week_data = prediction_reader.read_week(first_week)
            # Display the first 5 rows of the DataFrame.
            print(week_data.head())
    except Exception as e:
        # Catch and report any errors during prediction data reading.
        print(f"Could not read prediction data. Error: {e}")


    print("\n--- Reading Analytics Data ---")
    try:
        # Initialize DataReader for analytics data.
        analytics_reader = DataReader(analytics_data_dir)
        # Get list of weeks available in the analytics data.
        analytics_weeks = analytics_reader.get_weeks()
        print(f"Available weeks in analytics data: {analytics_weeks}")

        # Proceed if weeks are found.
        if analytics_weeks:
            # Select the first week found for demonstration.
            first_week = analytics_weeks[0]
            print(f"Reading data for week {first_week} from analytics data...")
            # Read the combined data for the selected week.
            week_data = analytics_reader.read_week(first_week)
            # Display the first 5 rows of the DataFrame.
            print(week_data.head())
    except Exception as e:
        # Catch and report any errors during analytics data reading.
        print(f"Could not read analytics data. Error: {e}")


if __name__ == '__main__':
    main()
