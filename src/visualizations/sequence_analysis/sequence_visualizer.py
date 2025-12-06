"""
Module for visualizing individual player sequences, comparing input trajectory,
expected final output position, and ball landing location.
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
    Handles the generation of sequence-level visualizations.
    """
    def __init__(self, data_reader, output_dir='visualizations/sequence_plots'):
        self.reader = data_reader
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_week(self, week, limit=None):
        """
        Generates sequence plots for a specific week.
        
        Args:
            week (str): The week identifier.
            limit (int, optional): Max sequences to process (for testing).
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
        # 'player_to_predict' column is boolean or 0/1.
        # Ensure compatible type comparison.
        if 'player_to_predict' in input_df.columns:
            # Handle string 'True'/'False' or boolean
            if input_df['player_to_predict'].dtype == 'object':
                 target_players = input_df[input_df['player_to_predict'].astype(str) == 'True']
            else:
                 target_players = input_df[input_df['player_to_predict'] == True]
        else:
            print("'player_to_predict' column not found in input.")
            return

        # Group by sequence identifier
        # A sequence is defined by game_id, play_id, nfl_id
        grouped = target_players.groupby(['game_id', 'play_id', 'nfl_id'])
        
        sequences = list(grouped.groups.keys())
        if limit:
            sequences = sequences[:limit]
            print(f"Processing first {limit} sequences...")
        else:
            print(f"Processing all {len(sequences)} sequences...")

        week_dir = os.path.join(self.output_dir, f'week_{week}')
        os.makedirs(week_dir, exist_ok=True)

        for (game_id, play_id, nfl_id) in tqdm(sequences, desc="Generating plots"):
            try:
                # 1. Get Input Trajectory
                # We use the group indices to fetch lines properly
                input_seq = grouped.get_group((game_id, play_id, nfl_id))
                
                # 2. Get Output Trajectory (to extract final position)
                output_seq = output_df[
                    (output_df['game_id'] == game_id) & 
                    (output_df['play_id'] == play_id) & 
                    (output_df['nfl_id'] == nfl_id)
                ]
                
                if output_seq.empty:
                    # Skip if no corresponding output (should usually exist)
                    continue

                # Get the FINAL position from output
                # Assuming simple sort by frame_id or just last row if ordered
                final_output_pos = output_seq.iloc[-1]
                
                # 3. Get Ball Landing Position
                # Assuming it's in the input sequence rows
                ball_land_x = input_seq['ball_land_x'].iloc[0]
                ball_land_y = input_seq['ball_land_y'].iloc[0]

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
        plt.figure(figsize=(12, 6))
        
        # Plot Field Context (Simplified)
        plt.xlim(0, 120)
        plt.ylim(0, 53.3)
        plt.grid(True, alpha=0.3)
        
        # 1. Input Trajectory
        plt.plot(input_df['x'], input_df['y'], 'b.-', label='Input Path', markersize=4, alpha=0.7)
        # Mark start of input
        plt.plot(input_df['x'].iloc[0], input_df['y'].iloc[0], 'bo', label='Input Start')
        # Mark end of input
        plt.plot(input_df['x'].iloc[-1], input_df['y'].iloc[-1], 'bs', label='Input End')

        # 2. Final Output Position
        plt.plot(output_final_row['x'], output_final_row['y'], 'rx', label='Target Final Pos', markersize=10, markeredgewidth=2)
        
        # 3. Ball Landing
        if pd.notna(ball_x) and pd.notna(ball_y):
            plt.plot(ball_x, ball_y, 'yo', label='Ball Landing', markersize=10, markeredgecolor='black')

        plt.title(f'Sequence: Game {game_id} Play {play_id} Player {nfl_id}')
        plt.xlabel('X (yards)')
        plt.ylabel('Y (yards)')
        plt.legend()
        
        # Add text info
        role = input_df['player_role'].iloc[0] if 'player_role' in input_df.columns else 'Unknown'
        plt.text(5, 50, f"Role: {role}", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        filename = f"seq_{game_id}_{play_id}_{nfl_id}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

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
        
        for week in tqdm(weeks, desc="Processing Weeks"):
            try:
                self.visualize_week(week, limit=limit_per_week)
            except Exception as e:
                print(f"Error processing week {week}: {e}")

def main():
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
        # We set a default limit of 20 per week to allow a quick demo run that verifies all weeks are processed.
        # To generate EVERY sequence (~7600 per week), change limit_per_week to None.
        visualizer.generate_all_weeks(limit_per_week=20)
        
        print(f"All visualizations saved to {visualizer.output_dir}")
        print("Tip: To generate ALL sequences (approx 130k images), edit the script and set limit_per_week=None in main()")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
