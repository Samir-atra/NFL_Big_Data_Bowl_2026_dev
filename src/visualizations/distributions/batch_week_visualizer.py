"""
Module for batch generating visualizations for ALL individual weeks.
This script iterates through all weeks and generates a set of 5 plots for EACH week,
saving them into separate subdirectories.
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
    Handles the generation of per-week data distribution visualizations for the entire dataset.
    """
    def __init__(self, data_reader, output_dir='visualizations/plots_batch_weeks'):
        """
        Initializes the BatchWeekVisualizer.

        Args:
            data_reader (DataReader): An instance of DataReader to fetch data.
            output_dir (str, optional): Directory to save the generated plots. 
        """
        self.reader = data_reader
        self.output_dir = output_dir
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all_weeks(self):
        """
        Iterates through all available weeks and generates visualizations for each.
        """
        weeks = self.reader.get_weeks()
        print(f"Found {len(weeks)} weeks of data. Starting batch generation...")
        
        for week in tqdm(weeks, desc="Processing Weeks"):
            try:
                self.process_week(week)
            except Exception as e:
                print(f"Error processing week {week}: {e}")

    def process_week(self, week):
        """
        Generates all plots for a single week and saves them in a week-specific folder.
        """
        # Create a subdirectory for each week
        week_output_dir = os.path.join(self.output_dir, f'week_{week}')
        os.makedirs(week_output_dir, exist_ok=True)

        df = self.reader.read_week(week)
        
        # Determine coordinate columns (Fix for x vs x_x issue)
        x_col = 'x' if 'x' in df.columns else ('x_x' if 'x_x' in df.columns else None)
        y_col = 'y' if 'y' in df.columns else ('y_x' if 'y_x' in df.columns else None)
        
        # 1. Height
        self.plot_histogram(df, 'player_height', week, week_output_dir, transform='height_inches')
        
        # 2. Weight
        self.plot_histogram(df, 'player_weight', week, week_output_dir)
        
        # 3. Speed by Position
        self.plot_boxplot(df, 's', week, week_output_dir, 'Speed (s)')
        
        # 4. Acceleration by Position
        self.plot_boxplot(df, 'a', week, week_output_dir, 'Acceleration (a)')
        
        # 5. Field Positions
        if x_col and y_col:
            self.plot_field_positions(df, x_col, y_col, week, week_output_dir)

    def plot_histogram(self, df, col, week, output_dir, transform=None):
        if col not in df.columns:
            return

        data = df[col]
        title_suffix = col.replace('player_', '').capitalize()
        xlabel = col
        
        # Handle transformation (e.g., height string to inches)
        if transform == 'height_inches':
            def convert_height(h):
                if pd.isna(h): return None
                try:
                    if isinstance(h, str) and '-' in h:
                        f, i = map(int, h.split('-'))
                        return f * 12 + i
                    return float(h)
                except: return None
            data = data.apply(convert_height)
            xlabel = 'Height (inches)'
        elif col == 'player_weight':
             xlabel = 'Weight (lbs)'

        plt.figure(figsize=(10, 6))
        sns.histplot(data.dropna(), kde=True, bins=20)
        plt.title(f'Distribution of {title_suffix} - Week {week}')
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'{title_suffix.lower()}_distribution_w{week}.png'))
        plt.close()

    def plot_boxplot(self, df, col, week, output_dir, ylabel):
        if 'player_position' not in df.columns or col not in df.columns:
            return
            
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='player_position', y=col, data=df)
        plt.title(f'Player {ylabel.split()[0]} by Position - Week {week}')
        plt.xlabel('Position')
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        filename = 'speed' if col == 's' else 'acceleration'
        plt.savefig(os.path.join(output_dir, f'{filename}_by_position_w{week}.png'))
        plt.close()

    def plot_field_positions(self, df, x_col, y_col, week, output_dir):
        plt.figure(figsize=(12, 6))
        hue = 'player_side' if 'player_side' in df.columns else None
        # Use full data for individual weeks as they are smaller than global
        sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue, alpha=0.5, s=10)
        plt.title(f'Player Positions on Field - Week {week}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'field_positions_w{week}.png'))
        plt.close()

def main():
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
