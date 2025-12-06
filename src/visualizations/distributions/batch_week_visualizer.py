"""
Module: Batch Week Visualizer
=============================

This module acts as a batch processor for generating individual weekly visualizations.
Unlike the Global visualizer which aggregates data, and the DataVisualizer which handles 
one week explicitly, this script iterates through the entire dataset and generates a 
complete set of 5 standard plots for *every single week* found.

Key Features:
-------------
- Automated Iteration: Finds all available weeks via DataReader and processes them sequentially.
- Organized Output: Creates a specific subdirectory for each week (e.g., `batch_week_plots/week_2023_w01`).
- Robustness: Handles column naming variations (x/y vs x_x/y_x) automatically.
- Progress Tracking: Uses tqdm to show progress through the weeks.

Usage:
------
Run clearly to populate the batch plots directory:
    $ python src/visualizations/distributions/batch_week_visualizer.py
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add the src directory to the system path to find the data_reader module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.data_reader import DataReader

class BatchWeekVisualizer:
    """
    Manager class for batch generation of weekly data visualizations.
    
    It automates the process of reading every week's data and producing a standardized
    packet of 5 plots (Height, Weight, Speed, Acceleration, Field Position) for each.
    """
    
    def __init__(self, data_reader, output_dir='visualizations/plots_batch_weeks'):
        """
        Initializes the BatchWeekVisualizer.

        Args:
            data_reader (DataReader): An instance of DataReader to fetch data.
            output_dir (str, optional): Root directory where week-specific subfolders 
                                        will be created. Defaults to 'visualizations/plots_batch_weeks'.
        """
        self.reader = data_reader
        self.output_dir = output_dir
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all_weeks(self):
        """
        Main driver method that iterates through all available weeks and triggers visualization.
        """
        weeks = self.reader.get_weeks()
        print(f"Found {len(weeks)} weeks of data. Starting batch generation...")
        
        # Iterate through weeks with a progress bar
        for week in tqdm(weeks, desc="Processing Weeks"):
            try:
                self.process_week(week)
            except Exception as e:
                print(f"Error processing week {week}: {e}")

    def process_week(self, week):
        """
        Generates the full suite of plots for a single week.
        
        This method:
        1. Creates the week string subdirectory.
        2. Loads the week's data.
        3. Identifies correct coordinate columns.
        4. Calls individual plotting methods.
        
        Args:
            week (str): The week identifier (e.g., '2023_w01').
        """
        # Create a subdirectory for each week
        week_output_dir = os.path.join(self.output_dir, f'week_{week}')
        os.makedirs(week_output_dir, exist_ok=True)

        # Load data
        df = self.reader.read_week(week)
        
        # Determine coordinate columns (Fix for x vs x_x issue)
        # Some weeks/files might have merged columns resulting in suffixes.
        x_col = 'x' if 'x' in df.columns else ('x_x' if 'x_x' in df.columns else None)
        y_col = 'y' if 'y' in df.columns else ('y_x' if 'y_x' in df.columns else None)
        
        # 1. Plot Height Distribution
        # Note: We apply a 'height_inches' transform here to standardize data
        self.plot_histogram(df, 'player_height', week, week_output_dir, transform='height_inches')
        
        # 2. Plot Weight Distribution
        self.plot_histogram(df, 'player_weight', week, week_output_dir)
        
        # 3. Plot Speed ('s') by Position
        self.plot_boxplot(df, 's', week, week_output_dir, 'Speed (s)')
        
        # 4. Plot Acceleration ('a') by Position
        self.plot_boxplot(df, 'a', week, week_output_dir, 'Acceleration (a)')
        
        # 5. Plot Field Positions (Scatter)
        if x_col and y_col:
            self.plot_field_positions(df, x_col, y_col, week, week_output_dir)

    def plot_histogram(self, df, col, week, output_dir, transform=None):
        """
        Generic method to plot a histogram for a given column.
        
        Args:
            df (pd.DataFrame): The data source.
            col (str): The column name to plot (e.g., 'player_height').
            week (str): Week identifier for the title.
            output_dir (str): Directory to save the image.
            transform (str, optional): helper flag to trigger specific data conversions 
                                       (e.g., 'height_inches').
        """
        if col not in df.columns:
            return

        data = df[col]
        
        # Clean up column name for title (remove prefix, capitalize)
        title_suffix = col.replace('player_', '').capitalize()
        xlabel = col
        
        # Handle specific data transformations
        if transform == 'height_inches':
            def convert_height(h):
                if pd.isna(h): return None
                try:
                    # Convert '6-2' to 74 inches
                    if isinstance(h, str) and '-' in h:
                        f, i = map(int, h.split('-'))
                        return f * 12 + i
                    return float(h)
                except: return None
            data = data.apply(convert_height)
            xlabel = 'Height (inches)'
        elif col == 'player_weight':
             xlabel = 'Weight (lbs)'

        # Generate Plot
        plt.figure(figsize=(10, 6))
        sns.histplot(data.dropna(), kde=True, bins=20)
        plt.title(f'Distribution of {title_suffix} - Week {week}')
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        
        # Save Plot
        plt.savefig(os.path.join(output_dir, f'{title_suffix.lower()}_distribution_w{week}.png'))
        plt.close()

    def plot_boxplot(self, df, col, week, output_dir, ylabel):
        """
        Generic method to plot a boxplot by player position.
        
        Args:
            df (pd.DataFrame): Data source.
            col (str): Column to visualize ('s' or 'a').
            week (str): Week identifier.
            output_dir (str): Save location.
            ylabel (str): Label for Y-axis (e.g., 'Speed (s)').
        """
        if 'player_position' not in df.columns or col not in df.columns:
            return
            
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='player_position', y=col, data=df)
        plt.title(f'Player {ylabel.split()[0]} by Position - Week {week}')
        plt.xlabel('Position')
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        
        # Determine filename prefix
        filename = 'speed' if col == 's' else 'acceleration'
        plt.savefig(os.path.join(output_dir, f'{filename}_by_position_w{week}.png'))
        plt.close()

    def plot_field_positions(self, df, x_col, y_col, week, output_dir):
        """
        Plots a scatter map of player locations.
        
        Args:
            df (pd.DataFrame): Data source.
            x_col (str): Column name for X coordinate.
            y_col (str): Column name for Y coordinate.
            week (str): Week identifier.
            output_dir (str): Save location.
        """
        plt.figure(figsize=(12, 6))
        
        hue = 'player_side' if 'player_side' in df.columns else None
        
        # We alpha blend to 0.5 to show density where players overlap
        sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue, alpha=0.5, s=10)
        
        plt.title(f'Player Positions on Field - Week {week}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'field_positions_w{week}.png'))
        plt.close()

def main():
    """
    Main execution block using the Analytics dataset.
    """
    # Using the Analytics path provided by the user
    analytics_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
    
    if not os.path.exists(analytics_data_dir):
        print(f"Directory not found: {analytics_data_dir}")
        return

    print(f"initializing batch visualizer for: {analytics_data_dir}")
    
    try:
        reader = DataReader(analytics_data_dir)
        # Output directory relative to this script
        output_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'batch_week_plots'))
        
        visualizer = BatchWeekVisualizer(reader, output_dir=output_base_dir)
        visualizer.generate_all_weeks()
        
        print(f"All batch visualizations saved to {visualizer.output_dir}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
