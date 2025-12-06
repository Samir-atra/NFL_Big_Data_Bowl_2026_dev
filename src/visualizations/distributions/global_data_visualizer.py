"""
Module: Global Data Visualizer
==============================

This module provides functionality to visualize data distributions across the ENTIRE dataset 
(all available weeks). It is designed to aggregate data from multiple tracking weeks to 
generate a comprehensive separate analysis.

Key Features:
-------------
- Aggregation: Combines data from all 18 weeks (approx. 5.4M records).
- Unique Player Analysis: For static attributes like Height and Weight, it deduplicates 
  players by NFL ID to ensure the distribution represents the population, not the 
  frequency of appearance in plays.
- Robustness: Automatically handles inconsistent column naming (e.g., 'x' vs 'x_x').
- Subsampling: Handles large dataset visualization by subsampling for scatter plots.

Usage:
------
Run this script directly to generate the global plots:
    $ python src/visualizations/distributions/global_data_visualizer.py
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # For progress bar

# Add the src directory to the system path to find the data_reader module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.data_reader import DataReader

class GlobalDataVisualizer:
    """
    Explorer class for generating global-level data distribution visualizations.
    
    This class is responsible for iterating over all available data weeks, attempting to 
    aggregate them into a single substantial DataFrame, and then producing summary 
    statistical plots.
    """
    
    def __init__(self, data_reader, output_dir='visualizations/plots_global'):
        """
        Initializes the GlobalDataVisualizer with a DataReader and output path.

        Args:
            data_reader (DataReader): An initialized DataReader instance to fetch data.
            output_dir (str, optional): The directory where generated plots will be saved. 
                                        Defaults to 'visualizations/plots_global'.
        """
        self.reader = data_reader
        self.output_dir = output_dir
        # Ensure the output directory exists to avoid IO errors later
        os.makedirs(self.output_dir, exist_ok=True)

    # Columns required for the specific plots we intend to generate.
    # We only load these to keep memory usage manageable when aggregating 5M+ rows.
    columns_needed = ['nfl_id', 'player_height', 'player_weight', 'player_position', 's', 'a', 'x', 'y', 'player_side']

    def plot_aggregated_stats(self):
        """
        Aggregates data from ALL available weeks and generates summary plots.
        
        This process involves:
        1. Iterating through all weeks found by the DataReader.
        2. Reading each week's data.
        3. Extracting only the necessary columns (handling scheman variations).
        4. Concatenating into a single master DataFrame.
        5. Generating plots based on this aggregated data.
        
        Note:
            This method is memory intensive. It uses `self.columns_needed` to minimize footprint.
        """
        weeks = self.reader.get_weeks()
        print(f"Aggregating data from {len(weeks)} weeks for global plots...")
        
        all_data_frames = []
        
        # tqdm provides a progress bar for the iteration
        for week in tqdm(weeks, desc="Loading Data"):
            try:
                # Read full week data using the DataReader
                df = self.reader.read_week(week)
                
                # Filter for existing columns only to avoid KeyErrors
                cols_to_keep = [c for c in self.columns_needed if c in df.columns]
                
                # Robustness Check: Handle potential merge suffixes (x_x vs x)
                # Sometimes data merging processes result in suffixed column names.
                if 'x' not in df.columns and 'x_x' in df.columns:
                    df['x'] = df['x_x']
                    cols_to_keep.append('x')
                if 'y' not in df.columns and 'y_x' in df.columns:
                    df['y'] = df['y_x']
                    cols_to_keep.append('y')
                    
                # Append the subset DataFrame to our list
                if not df.empty:
                    all_data_frames.append(df[cols_to_keep])
                    
            except Exception as e:
                # Log errors but continue processing other weeks
                print(f"Skipping week {week} due to error: {e}")

        if not all_data_frames:
            print("No data found to aggregate.")
            return

        print("Concatenating data (this may take a moment)...")
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        print(f"Total records aggregated: {len(combined_df)}")

        # Create a specific subdirectory for the aggregated plots
        agg_output_dir = os.path.join(self.output_dir, 'aggregated_all_weeks')
        os.makedirs(agg_output_dir, exist_ok=True)

        print("Generating aggregated plots...")
        
        # --- Plot Generation Logic ---

        # 1. Player Attributes (Height/Weight)
        # CRITICAL: We drop duplicates by NFL ID. If we didn't, a player who plays 1000 frames
        # would be counted 1000 times, skewing the distribution towards starters.
        # We want the distribution of *unique players*.
        if 'nfl_id' in combined_df.columns:
            unique_players = combined_df.drop_duplicates(subset=['nfl_id'])
            self.plot_player_attributes(unique_players, "ALL_WEEKS_UNIQUE", agg_output_dir)
        else:
            # Fallback if no ID is present (unlikely in this dataset)
            self.plot_player_attributes(combined_df, "ALL_WEEKS", agg_output_dir)

        # 2. Dynamics (Speed/Acceleration)
        # For these, we WANT all frames, as we are looking at the distribution of movement samples.
        self.plot_player_speed_acceleration(combined_df, "ALL_WEEKS", agg_output_dir)

        # 3. Field Positions
        # If the dataset is too large (> 500k), scatter plots become slow and unreadable (overplotting).
        # We subsample to 500k points to maintain performance and visual utility.
        if len(combined_df) > 500000:
            print("Subsampling position data for scatter plot clarity...")
            sampled_df = combined_df.sample(n=500000, random_state=42)
            self.plot_field_positions(sampled_df, "ALL_WEEKS", agg_output_dir)
        else:
            self.plot_field_positions(combined_df, "ALL_WEEKS", agg_output_dir)

    def plot_all_weeks(self):
        """
        Iterates through all available weeks and generates SEPARATE visualizations for each.
        
        Use this if you want 18 separate folders of plots (one per week) rather than one aggregate.
        """
        weeks = self.reader.get_weeks()
        print(f"Found {len(weeks)} weeks of data.")
        
        for week in tqdm(weeks, desc="Processing weeks"):
            try:
                self.process_week(week)
            except Exception as e:
                print(f"Error processing week {week}: {e}")

    def process_week(self, week):
        """
        Helper method to generate all plots for a single week instance.
        
        Args:
            week (str): The week identifier (e.g., '2023_w01').
        """
        # Create a subdirectory for each week to keep output organized
        week_output_dir = os.path.join(self.output_dir, f'week_{week}')
        os.makedirs(week_output_dir, exist_ok=True)

        df = self.reader.read_week(week)
        
        # Generate the standard suite of plots
        self.plot_player_attributes(df, week, week_output_dir)
        self.plot_player_speed_acceleration(df, week, week_output_dir)
        self.plot_field_positions(df, week, week_output_dir)

    def plot_player_attributes(self, df, week, output_dir):
        """
        Generates and saves histograms for player height and weight.
        
        Args:
            df (pd.DataFrame): Data containing 'player_height' and 'player_weight'.
            week (str): Label for the plot title (e.g., '2023_w01' or 'ALL_WEEKS').
            output_dir (str): Path to save the PNG files.
        """
        
        # Helper function to convert height string 'feet-inches' (e.g., '6-2') into inches.
        def convert_height_to_inches(height_str):
            if pd.isna(height_str):
                return None
            try:
                # Check for standard 'F-I' format
                if isinstance(height_str, str) and '-' in height_str:
                    feet, inches = map(int, height_str.split('-'))
                    return (feet * 12) + inches
                # Handle cases where it might already be numeric
                return float(height_str)
            except (ValueError, TypeError):
                return None
        
        # Plot Height Distribution
        if 'player_height' in df.columns:
            # Create copy to avoid SettingWithCopyWarning on the original dataframe
            df = df.copy()
            df['player_height_inches'] = df['player_height'].apply(convert_height_to_inches)

            plt.figure(figsize=(10, 6))
            sns.histplot(df['player_height_inches'].dropna(), kde=True, bins=20)
            plt.title(f'Distribution of Player Height - {week}')
            plt.xlabel('Height (inches)')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f'height_distribution_w{week}.png'))
            plt.close()

        # Plot Weight Distribution
        if 'player_weight' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['player_weight'].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of Player Weight - {week}')
            plt.xlabel('Weight (lbs)')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f'weight_distribution_w{week}.png'))
            plt.close()

    def plot_player_speed_acceleration(self, df, week, output_dir):
        """
        Generates and saves box plots for player speed (s) and acceleration (a).
        
        Args:
            df (pd.DataFrame): Data containing 's', 'a', and 'player_position'.
            week (str): Label for the plot title.
            output_dir (str): Path to save the PNG files.
        """
        
        # Plot Speed by Position
        if 'player_position' in df.columns and 's' in df.columns:
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='player_position', y='s', data=df)
            plt.title(f'Player Speed by Position - {week}')
            plt.xlabel('Position')
            plt.ylabel('Speed (s)')
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(output_dir, f'speed_by_position_w{week}.png'))
            plt.close()

        # Plot Acceleration by Position
        if 'player_position' in df.columns and 'a' in df.columns:
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='player_position', y='a', data=df)
            plt.title(f'Player Acceleration by Position - {week}')
            plt.xlabel('Position')
            plt.ylabel('Acceleration (a)')
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(output_dir, f'acceleration_by_position_w{week}.png'))
            plt.close()

    def plot_field_positions(self, df, week, output_dir):
        """
        Generates and saves a scatter plot of player xy coordinates.
        
        Args:
            df (pd.DataFrame): Data containing 'x', 'y' (or suffixed versions).
            week (str): Label for the plot title.
            output_dir (str): Path to save the PNG files.
        """
        
        # Robustly determine column names for coordinates
        x_col = 'x' if 'x' in df.columns else None
        y_col = 'y' if 'y' in df.columns else None
        
        if x_col and y_col:
            plt.figure(figsize=(12, 6)) # Standard football field aspect ratio approx
            
            # Color points by side (Offense/Defense) if available
            hue = 'player_side' if 'player_side' in df.columns else None
            
            sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue, alpha=0.3, s=10)
            plt.title(f'Player Positions on Field - {week}')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            # Position legend outside the plot area
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'field_positions_w{week}.png'))
            plt.close()

def main():
    """
    Main entry point for the Global Data Visualizer.
    Configured to run on the Analytics dataset by default.
    """
    prediction_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
    
    if not os.path.exists(prediction_data_dir):
        print(f"Directory not found: {prediction_data_dir}")
        return

    print(f"initializing visualizer for: {prediction_data_dir}")
    
    try:
        reader = DataReader(prediction_data_dir)
        # Output directory relative to this script
        output_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'global_plots'))
        
        visualizer = GlobalDataVisualizer(reader, output_dir=output_base_dir)
        
        # Option 1: Generate per-week plots (Iterates all weeks but keeps them separate)
        # visualizer.plot_all_weeks()
        
        # Option 2: Generate aggregated plots (Combines all data into one view)
        # This is the default action for the global visualizer
        visualizer.plot_aggregated_stats()
        
        print(f"All visualizations saved to {visualizer.output_dir}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
