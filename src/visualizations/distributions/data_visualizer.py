"""
Module for visualizing static player attributes and dynamic movement data distributions.

This script uses the Matplotlib and Seaborn libraries to generate plots such as
player height/weight distributions, speed/acceleration by position, and scatter
plots of on-field player coordinates. It relies on the DataReader module to load
data for specific weeks.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the system path to find the data_reader module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_reader import DataReader

class DataVisualizer:
    """
    Handles the generation and saving of various data distribution visualizations
    for NFL Big Data Bowl data.
    """
    def __init__(self, data_reader, output_dir='visualizations/plots'):
        """
        Initializes the DataVisualizer.

        Args:
            data_reader (DataReader): An instance of DataReader to fetch data.
            output_dir (str, optional): Directory to save the generated plots. 
                                        Defaults to 'visualizations/plots'.
        """
        self.reader = data_reader
        self.output_dir = output_dir
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_player_attributes(self, week):
        """
        Generates and saves histograms for player height and weight distributions.

        Args:
            week (str): The specific week (e.g., '2023_w01') to plot data for.
        """
        df = self.reader.read_week(week)
        
        # Helper function to convert height string 'feet-inches' to total inches
        def convert_height_to_inches(height_str):
            if pd.isna(height_str):
                return None
            try:
                feet, inches = map(int, height_str.split('-'))
                return (feet * 12) + inches
            except ValueError:
                return None
        
        # Apply conversion
        df['player_height_inches'] = df['player_height'].apply(convert_height_to_inches)

        # Plot 1: Player Height Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(df['player_height_inches'].dropna(), kde=True, bins=20)
        plt.title(f'Distribution of Player Height - Week {week}')
        plt.xlabel('Height (inches)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.output_dir, f'height_distribution_w{week}.png'))
        plt.close()

        # Plot 2: Player Weight Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(df['player_weight'], kde=True, bins=30)
        plt.title(f'Distribution of Player Weight - Week {week}')
        plt.xlabel('Weight (lbs)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.output_dir, f'weight_distribution_w{week}.png'))
        plt.close()

    def plot_player_speed_acceleration(self, week):
        """
        Generates and saves box plots for player speed ('s') and acceleration ('a')
        grouped by their position.

        Args:
            week (str): The specific week (e.g., '2023_w01') to plot data for.
        """
        df = self.reader.read_week(week)

        # Plot 1: Speed by Position
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='player_position', y='s', data=df)
        plt.title(f'Player Speed by Position - Week {week}')
        plt.xlabel('Position')
        plt.ylabel('Speed (s)')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.output_dir, f'speed_by_position_w{week}.png'))
        plt.close()

        # Plot 2: Acceleration by Position
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='player_position', y='a', data=df)
        plt.title(f'Player Acceleration by Position - Week {week}')
        plt.xlabel('Position')
        plt.ylabel('Acceleration (a)')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.output_dir, f'acceleration_by_position_w{week}.png'))
        plt.close()

    def plot_field_positions(self, week):
        """
        Generates and saves a scatter plot showing player (x, y) coordinates on the field,
        colored by the 'player_side' (offense/defense).

        Args:
            week (str): The specific week (e.g., '2023_w01') to plot data for.
        """
        df = self.reader.read_week(week)
        
        # Scatter plot of all player positions on the field
        plt.figure(figsize=(10, 10))
        # Note: Assuming 'x_x' and 'y_x' are the field coordinates after join/read
        sns.scatterplot(x='x_x', y='y_x', data=df, hue='player_side', alpha=0.5)
        plt.title(f'Player Positions on Field - Week {week}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.savefig(os.path.join(self.output_dir, f'field_positions_w{week}.png'))
        plt.close()


def main():
    """
    Main execution function to set up the DataReader and generate a set of
    visualizations for a sample week of prediction data.
    """
    prediction_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
    
    try:
        print("--- Generating Visualizations for Prediction Data ---")
        # Initialize DataReader for the prediction data
        prediction_reader = DataReader(prediction_data_dir)
        # Define output directory relative to the script location
        output_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'prediction_plots'))
        visualizer = DataVisualizer(prediction_reader, output_dir=output_base_dir)
        
        # Using week 1 for visualization
        week_to_visualize = '2023_w01'
        
        print(f"Generating player attribute plots for week {week_to_visualize}...")
        visualizer.plot_player_attributes(week_to_visualize)
        
        print(f"Generating player speed and acceleration plots for week {week_to_visualize}...")
        visualizer.plot_player_speed_acceleration(week_to_visualize)
        
        print(f"Generating field position plots for week {week_to_visualize}...")
        visualizer.plot_field_positions(week_to_visualize)

        print(f"Visualizations saved to {visualizer.output_dir}")

    except Exception as e:
        print(f"Could not generate visualizations. Error: {e}")


if __name__ == '__main__':
    main()
