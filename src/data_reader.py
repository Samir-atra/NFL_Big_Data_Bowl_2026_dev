import pandas as pd
import os
import glob

class DataReader:
    """
    A class to read the NFL Big Data Bowl 2026 data.
    """
    def __init__(self, data_dir):
        """
        Initializes the DataReader with the path to the data directory.
        """
        self.data_dir = data_dir

    def get_weeks(self):
        """
        Gets a list of the available weeks in the data directory.
        """
        weeks = set()
        # Use glob to find all csv files in the directory
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        
        for f_path in csv_files:
            filename = os.path.basename(f_path)
            # Distinguish between input and output files to get the week identifier
            if filename.startswith('input_'):
                week_id = filename.replace('input_', '').replace('.csv', '')
                weeks.add(week_id)
        return sorted(list(weeks))

    def read_input(self, week):
        """
        Reads the input data for a given week.
        """
        path = os.path.join(self.data_dir, f'input_{week}.csv')
        return pd.read_csv(path)

    def read_output(self, week):
        """
        Reads the output data for a given week.
        """
        path = os.path.join(self.data_dir, f"output_{week}.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    def read_week(self, week):
        """
        Reads and merges the input and output data for a given week.
        """
        input_df = self.read_input(week)
        output_df = self.read_output(week) 
        if output_df is not None:
            # A simple concatenation is more robust as input and output files have distinct frame ranges.
            # Frame 0 is in input, frames > 0 are in output.
            return pd.concat([input_df, output_df], ignore_index=True)
        return input_df

def main():
    """
    Main function to demonstrate the DataReader class.
    """
    # Path to the prediction and analytics data
    prediction_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    analytics_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/'

    print("--- Reading Prediction Data ---")
    try:
        prediction_reader = DataReader(prediction_data_dir)
        prediction_weeks = prediction_reader.get_weeks()
        print(f"Available weeks in prediction data: {prediction_weeks}")

        if prediction_weeks:
            # Read and print the head of the first week's data
            first_week = prediction_weeks[0]
            print(f"Reading data for week {first_week} from prediction data...")
            week_data = prediction_reader.read_week(first_week)
            print(week_data.head())
    except Exception as e:
        print(f"Could not read prediction data. Error: {e}")


    print("\n--- Reading Analytics Data ---")
    try:
        analytics_reader = DataReader(analytics_data_dir)
        analytics_weeks = analytics_reader.get_weeks()
        print(f"Available weeks in analytics data: {analytics_weeks}")

        if analytics_weeks:
            # Read and print the head of the first week's data
            first_week = analytics_weeks[0]
            print(f"Reading data for week {first_week} from analytics data...")
            week_data = analytics_reader.read_week(first_week)
            print(week_data.head())
    except Exception as e:
        print(f"Could not read analytics data. Error: {e}")


if __name__ == '__main__':
    main()
