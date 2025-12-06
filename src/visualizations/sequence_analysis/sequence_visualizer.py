"""
Module: Sequence Visualizer
===========================

This module is responsible for visualizing individual player movement sequences in detail.
Unlike the distribution visualizers which aggregate data, this script looks at the 
tracking data for specific (Game, Play, Player) tuples.

Key Features:
-------------
- Trajectory Plotting: Traces the path of a player from the input frames (blue).
- Target Verification: Plots the expected final position from the output data (red X).
- Contextual Elements: Shows the ball landing position (yellow dot) relative to the player.
- Comparison: Helps verify if the player's movement aligns with the play's outcome.

Usage:
------
Run to generate sequence plots (default limt 20 per week for demonstration):
    $ python src/visualizations/sequence_analysis/sequence_visualizer.py
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add the src directory to the system path to find the data_reader module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data_reader import DataReader

class SequenceVisualizer:
    """
    Handles the generation of sequence-level tracking visualizations.
    
    It serves as a tool for inspecting specific plays, debugging model inputs/outputs, 
    and understanding player movement relative to the ball.
    """
    
    def __init__(self, data_reader, output_dir='visualizations/sequence_plots'):
        """
        Initializes the SequenceVisualizer.

        Args:
            data_reader (DataReader): Initialized DataReader.
            output_dir (str, optional): Directory to save plots. 
                                        Defaults to 'visualizations/sequence_plots'.
        """
        self.reader = data_reader
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all_weeks(self, limit_per_week=20):
        """
        Iterates through all available weeks and generates sequence plots for each.
        
        Args:
            limit_per_week (int, optional): Max sequences to process per week. 
                                            Set to None to process ALL sequences (warning: large output).
        """
        weeks = self.reader.get_weeks()
        if not weeks:
            print("No weeks found in data directory.")
            return
            
        print(f"Found {len(weeks)} weeks of data. Starting batch sequence generation...")
        print(f"Limit per week: {limit_per_week if limit_per_week else 'ALL'}")
        
        # Iterate over all weeks with progress bar
        for week in tqdm(weeks, desc="Processing Weeks"):
            try:
                self.visualize_week(week, limit=limit_per_week)
            except Exception as e:
                print(f"Error processing week {week}: {e}")

    def visualize_week(self, week, limit=None):
        """
        Generates sequence plots for a specific week.
        
        Args:
            week (str): The week identifier (e.g., '2023_w01').
            limit (int, optional): Max sequences to process. Useful for quick testing.
        """
        print(f"Loading data for week {week}...")
        try:
            input_df = self.reader.read_input(week)
            output_df = self.reader.read_output(week)
        except Exception as e:
            print(f"Error reading data for week {week}: {e}")
            return

        if output_df is None:
            print(f"No output data found for week {week}.")
            return

        # Filter for players to predict
        # The 'player_to_predict' column indicates which players are the targets for the competition.
        if 'player_to_predict' in input_df.columns:
            # Handle potential type mismatches (string vs boolean)
            if input_df['player_to_predict'].dtype == 'object':
                 target_players = input_df[input_df['player_to_predict'].astype(str) == 'True']
            else:
                 target_players = input_df[input_df['player_to_predict'] == True]
        else:
            print("'player_to_predict' column not found in input.")
            return

        # Group by sequence identifier
        # A unique sequence is defined by (Game ID, Play ID, NFL ID)
        grouped = target_players.groupby(['game_id', 'play_id', 'nfl_id'])
        
        sequences = list(grouped.groups.keys())
        
        # Apply the limit if set
        if limit:
            sequences = sequences[:limit]
            print(f"Processing first {limit} sequences...")
        else:
            print(f"Processing all {len(sequences)} sequences...")

        # Create week-specific output directory
        week_dir = os.path.join(self.output_dir, f'week_{week}')
        os.makedirs(week_dir, exist_ok=True)

        for (game_id, play_id, nfl_id) in tqdm(sequences, desc="Generating plots"):
            try:
                # 1. Get Input Trajectory (Blue)
                input_seq = grouped.get_group((game_id, play_id, nfl_id))
                
                # 2. Get Output Trajectory (Red)
                # We need the output file to find the True Final Position
                output_seq = output_df[
                    (output_df['game_id'] == game_id) & 
                    (output_df['play_id'] == play_id) & 
                    (output_df['nfl_id'] == nfl_id)
                ]
                
                if output_seq.empty:
                    continue

                # The last frame of the output sequence is the final target position
                final_output_pos = output_seq.iloc[-1]
                
                # 3. Get Ball Landing Position (Yellow)
                # The ball landing info is repeated in the input rows
                ball_land_x = input_seq['ball_land_x'].iloc[0]
                ball_land_y = input_seq['ball_land_y'].iloc[0]

                # Generate the specific plot
                self.plot_sequence(
                    input_seq, 
                    final_output_pos, 
                    ball_land_x, ball_land_y,
                    game_id, play_id, nfl_id,
                    week_dir
                )

            except Exception as e:
                print(f"Error plotting sequence {game_id}-{play_id}-{nfl_id}: {e}")

    def plot_sequence(self, input_df, output_final_row, ball_x, ball_y, game_id, play_id, nfl_id, output_dir):
        """
        Creates and saves a single Matplotlib figure for the sequence.
        
        Args:
            input_df (pd.DataFrame): The input frames for the player.
            output_final_row (pd.Series): The row containing final position coordinates.
            ball_x (float): Ball landing X coordinate.
            ball_y (float): Ball landing Y coordinate.
            game_id (int): Game identifier.
            play_id (int): Play identifier.
            nfl_id (int): Player identifier.
            output_dir (str): Directory to save the PNG.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot Field Context (Simplified 120x53.3 yards)
        plt.xlim(0, 120)
        plt.ylim(0, 53.3)
        plt.grid(True, alpha=0.3)
        
        # 1. Input Trajectory
        # Plot the path with a line and markers for each frame
        plt.plot(input_df['x'], input_df['y'], 'b.-', label='Input Path', markersize=4, alpha=0.7)
        # Mark Start (Circle)
        plt.plot(input_df['x'].iloc[0], input_df['y'].iloc[0], 'bo', label='Input Start')
        # Mark End (Square)
        plt.plot(input_df['x'].iloc[-1], input_df['y'].iloc[-1], 'bs', label='Input End')

        # 2. Final Output Position (Red X)
        plt.plot(output_final_row['x'], output_final_row['y'], 'rx', label='Target Final Pos', markersize=10, markeredgewidth=2)
        
        # 3. Ball Landing (Yellow Circle)
        if pd.notna(ball_x) and pd.notna(ball_y):
            plt.plot(ball_x, ball_y, 'yo', label='Ball Landing', markersize=10, markeredgecolor='black')

        plt.title(f'Sequence: Game {game_id} Play {play_id} Player {nfl_id}')
        plt.xlabel('X (yards)')
        plt.ylabel('Y (yards)')
        plt.legend()
        
        # Add player role info if available
        role = input_df['player_role'].iloc[0] if 'player_role' in input_df.columns else 'Unknown'
        plt.text(5, 50, f"Role: {role}", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        filename = f"seq_{game_id}_{play_id}_{nfl_id}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def main():
    """
    Main entry point.
    """
    analytics_data_dir = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
    
    if not os.path.exists(analytics_data_dir):
        print(f"Directory not found: {analytics_data_dir}")
        return

    print(f"Initializing Sequence Visualizer for: {analytics_data_dir}")
    
    try:
        reader = DataReader(analytics_data_dir)
        output_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sequence_plots'))
        
        visualizer = SequenceVisualizer(reader, output_dir=output_base_dir)
        
        # Generate for ALL weeks
        # limit_per_week=20  -> Fast Demo (generates ~360 images total)
        # limit_per_week=None -> FULL RUN (generates ~130k images total)
        visualizer.generate_all_weeks(limit_per_week=20)
        
        print(f"All visualizations saved to {visualizer.output_dir}")
        print("Tip: To generate ALL sequences (approx 130k images), edit the script and set limit_per_week=None in main()")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
