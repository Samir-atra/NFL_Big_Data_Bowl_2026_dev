import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the system path to find the data_reader module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_reader import DataReader

class DataVisualizer:
    def __init__(self, data_reader, output_dir='visualizations/plots'):
        self.reader = data_reader
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_player_attributes(self, week):
        df = self.reader.read_week(week)
        
        # Convert height from 'feet-inches' to inches for plotting
        def convert_height_to_inches(height_str):
            if pd.isna(height_str):
                return None
            try:
                feet, inches = map(int, height_str.split('-'))
                return (feet * 12) + inches
            except ValueError:
                return None
        
        df['player_height_inches'] = df['player_height'].apply(convert_height_to_inches)

        plt.figure(figsize=(12, 6))
        sns.histplot(df['player_height_inches'].dropna(), kde=True, bins=20)
        plt.title(f'Distribution of Player Height - Week {week}')
        plt.xlabel('Height (inches)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.output_dir, f'height_distribution_w{week}.png'))
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.histplot(df['player_weight'], kde=True, bins=30)
        plt.title(f'Distribution of Player Weight - Week {week}')
        plt.xlabel('Weight (lbs)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.output_dir, f'weight_distribution_w{week}.png'))
        plt.close()

    def plot_player_speed_acceleration(self, week):
        df = self.reader.read_week(week)

        plt.figure(figsize=(15, 8))
        sns.boxplot(x='player_position', y='s', data=df)
        plt.title(f'Player Speed by Position - Week {week}')
        plt.xlabel('Position')
        plt.ylabel('Speed (s)')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.output_dir, f'speed_by_position_w{week}.png'))
        plt.close()

        plt.figure(figsize=(15, 8))
        sns.boxplot(x='player_position', y='a', data=df)
        plt.title(f'Player Acceleration by Position - Week {week}')
        plt.xlabel('Position')
        plt.ylabel('Acceleration (a)')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.output_dir, f'acceleration_by_position_w{week}.png'))
        plt.close()

    def plot_field_positions(self, week):
        df = self.reader.read_week(week)
        
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x='x_x', y='y_x', data=df, hue='player_side', alpha=0.5)
        plt.title(f'Player Positions on Field - Week {week}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.savefig(os.path.join(self.output_dir, f'field_positions_w{week}.png'))
        plt.close()


def main():
    prediction_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/'
    
    try:
        print("--- Generating Visualizations for Prediction Data ---")
        prediction_reader = DataReader(prediction_data_dir)
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
