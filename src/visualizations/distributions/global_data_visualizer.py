"""
Module for visualizing distributions across ALL available weeks of data.
This script iterates through all weeks found by DataReader, generating visualizations
for each, and fixing column name discrepancies found in previous versions.
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
    Handles the generation of data distribution visualizations for all available weeks
    in the NFL Big Data Bowl dataset.
    """
    def __init__(self, data_reader, output_dir='visualizations/plots_global'):
        """
        Initializes the GlobalDataVisualizer.

        Args:
            data_reader (DataReader): An instance of DataReader to fetch data.
            output_dir (str, optional): Directory to save the generated plots. 
                                        Defaults to 'visualizations/plots_global'.
        """
        self.reader = data_reader
        self.output_dir = output_dir
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    columns_needed = ['nfl_id', 'player_height', 'player_weight', 'player_position', 's', 'a', 'x', 'y', 'player_side']

    def plot_aggregated_stats(self):
        """
        Aggregates data from ALL available weeks and generates 5 summary plots.
        Uses memory-efficient loading by selecting only necessary columns.
        """
        weeks = self.reader.get_weeks()
        print(f"Aggregating data from {len(weeks)} weeks for global plots...")
        
        all_data_frames = []
        
        for week in tqdm(weeks, desc="Loading Data"):
            try:
                # Read full week data
                df = self.reader.read_week(week)
                
                # Filter for existing columns only
                cols_to_keep = [c for c in self.columns_needed if c in df.columns]
                
                # Check for x/y coordinate mapping if standard names don't exist
                if 'x' not in df.columns and 'x_x' in df.columns:
                    df['x'] = df['x_x']
                    cols_to_keep.append('x')
                if 'y' not in df.columns and 'y_x' in df.columns:
                    df['y'] = df['y_x']
                    cols_to_keep.append('y')
                    
                # Append subset to list
                if not df.empty:
                    all_data_frames.append(df[cols_to_keep])
                    
            except Exception as e:
                print(f"Skipping week {week} due to error: {e}")

        if not all_data_frames:
            print("No data found to aggregate.")
            return

        print("Concatenating data (this may take a moment)...")
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        print(f"Total records aggregated: {len(combined_df)}")

        # Create aggregate output directory
        agg_output_dir = os.path.join(self.output_dir, 'aggregated_all_weeks')
        os.makedirs(agg_output_dir, exist_ok=True)

        print("Generating aggregated plots...")
        
        # 1. Player Attributes (Height/Weight) - Unique Players Only
        # We drop duplicates by ID to avoid weighting by number of frames played
        if 'nfl_id' in combined_df.columns:
            unique_players = combined_df.drop_duplicates(subset=['nfl_id'])
            self.plot_player_attributes(unique_players, "ALL_WEEKS_UNIQUE", agg_output_dir)
        else:
            # Fallback if no ID
            self.plot_player_attributes(combined_df, "ALL_WEEKS", agg_output_dir)

        # 2. Dynamics (Speed/Acceleration) - All Frames
        self.plot_player_speed_acceleration(combined_df, "ALL_WEEKS", agg_output_dir)

        # 3. Field Positions - All Frames (might be heavy scatter plot, sample if too large?)
        # If > 1M points, maybe sample for visualization clarity
        if len(combined_df) > 500000:
            print("Subsampling position data for scatter plot clarity...")
            sampled_df = combined_df.sample(n=500000, random_state=42)
            self.plot_field_positions(sampled_df, "ALL_WEEKS", agg_output_dir)
        else:
            self.plot_field_positions(combined_df, "ALL_WEEKS", agg_output_dir)

    def plot_all_weeks(self):
        """
        Iterates through all available weeks and generates visualizations for each.
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
        Generates all plots for a single week.
        
        Args:
            week (str): The week identifier.
        """
        # Create a subdirectory for each week to keep things organized
        week_output_dir = os.path.join(self.output_dir, f'week_{week}')
        os.makedirs(week_output_dir, exist_ok=True)

        df = self.reader.read_week(week)
        
        self.plot_player_attributes(df, week, week_output_dir)
        self.plot_player_speed_acceleration(df, week, week_output_dir)
        self.plot_field_positions(df, week, week_output_dir)

    def plot_player_attributes(self, df, week, output_dir):
        """Generates histograms for player height and weight."""
        
        # Helper function to convert height string 'feet-inches' to total inches
        def convert_height_to_inches(height_str):
            if pd.isna(height_str):
                return None
            try:
                # Handle cases where it might already be a number or different format
                if isinstance(height_str, str) and '-' in height_str:
                    feet, inches = map(int, height_str.split('-'))
                    return (feet * 12) + inches
                return float(height_str) # Assume it's inches if not string with dash
            except (ValueError, TypeError):
                return None
        
        # Apply conversion safely
        if 'player_height' in df.columns:
            # Avoid SettingWithCopyWarning
            df = df.copy()
            df['player_height_inches'] = df['player_height'].apply(convert_height_to_inches)

            # Plot Height
            plt.figure(figsize=(10, 6))
            sns.histplot(df['player_height_inches'].dropna(), kde=True, bins=20)
            plt.title(f'Distribution of Player Height - {week}')
            plt.xlabel('Height (inches)')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f'height_distribution_w{week}.png'))
            plt.close()

        # Plot Weight
        if 'player_weight' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['player_weight'].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of Player Weight - {week}')
            plt.xlabel('Weight (lbs)')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f'weight_distribution_w{week}.png'))
            plt.close()

    def plot_player_speed_acceleration(self, df, week, output_dir):
        """Generates box plots for player speed and acceleration."""
        
        if 'player_position' in df.columns and 's' in df.columns:
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='player_position', y='s', data=df)
            plt.title(f'Player Speed by Position - {week}')
            plt.xlabel('Position')
            plt.ylabel('Speed (s)')
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(output_dir, f'speed_by_position_w{week}.png'))
            plt.close()

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
        """Generates scatter plot of player field positions."""
        
        # Check for 'x' and 'y' columns (fixed from 'x_x', 'y_x')
        x_col = 'x' if 'x' in df.columns else None
        y_col = 'y' if 'y' in df.columns else None
        
        if x_col and y_col:
            plt.figure(figsize=(12, 6)) # Field is rectangular
            hue = 'player_side' if 'player_side' in df.columns else None
            sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue, alpha=0.3, s=10)
            plt.title(f'Player Positions on Field - {week}')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'field_positions_w{week}.png'))
            plt.close()

def main():
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
        
        # Option 1: Generate per-week plots (commented out per user request focus)
        # visualizer.plot_all_weeks()
        
        # Option 2: Generate aggregated plots (User request)
        visualizer.plot_aggregated_stats()
        
        print(f"All visualizations saved to {visualizer.output_dir}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
