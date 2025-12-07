"""Sequence visualization script organized by player position.

This script generates per-week trajectory plots for each position,
showing player motion paths, final predicted positions, and ball landing locations.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure backend acts non-interactively
import matplotlib
matplotlib.use('Agg')


class SequenceVisualizerByPosition:
    """Handles the generation of sequence-level tracking visualizations,
    organized by Player Position.
    """

    def __init__(self, data_dir, output_dir):
        """Initialize the visualizer.

        Args:
            data_dir: Directory containing input_*.csv and output_*.csv files.
            output_dir: Directory to save generated plots.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_weeks(self):
        """Discover available weeks from input files.

        Returns:
            Sorted list of week identifiers (e.g., ['2023_w01', '2023_w02', ...]).
        """
        input_files = glob.glob(os.path.join(self.data_dir, "input_*.csv"))
        weeks = []
        for f in input_files:
            week_name = os.path.basename(f).replace('input_', '').replace('.csv', '')
            weeks.append(week_name)
        return sorted(weeks)

    def read_input(self, week):
        """Read input CSV for a specific week.

        Args:
            week: Week identifier string.

        Returns:
            DataFrame with input data.
        """
        file_path = os.path.join(self.data_dir, f"input_{week}.csv")
        return pd.read_csv(file_path)

    def read_output(self, week):
        """Read output CSV for a specific week.

        Args:
            week: Week identifier string.

        Returns:
            DataFrame with output data, or None if file doesn't exist.
        """
        file_path = os.path.join(self.data_dir, f"output_{week}.csv")
        if not os.path.exists(file_path):
            return None
        return pd.read_csv(file_path)

    def generate_all_weeks(self, limit_per_position=5):
        """Iterate through all available weeks and generate sequence plots.

        Args:
            limit_per_position: Max sequences to process per position per week.
        """
        weeks = self.get_weeks()
        if not weeks:
            print("No weeks found in data directory.")
            return

        print(f"Found {len(weeks)} weeks of data. Starting sequence generation...")
        print(f"Limit per position: {limit_per_position}")

        for week in tqdm(weeks, desc="Processing Weeks"):
            try:
                self.visualize_week(week, limit_per_pos=limit_per_position)
            except Exception as e:
                print(f"Error processing week {week}: {e}")

    def visualize_week(self, week, limit_per_pos=5):
        """Generate sequence plots for a specific week, grouping by position.

        Args:
            week: Week identifier string.
            limit_per_pos: Max sequences per position.
        """
        try:
            input_df = self.read_input(week)
            output_df = self.read_output(week)
        except Exception as e:
            print(f"Error reading data for week {week}: {e}")
            return

        if output_df is None:
            print(f"No output file found for week {week}, skipping.")
            return

        # Filter for players to predict
        if 'player_to_predict' not in input_df.columns:
            print("'player_to_predict' column not found.")
            return

        # Handle string/bool conversion
        if input_df['player_to_predict'].dtype == 'object':
            target_players = input_df[
                input_df['player_to_predict'].astype(str) == 'True'
            ]
        else:
            target_players = input_df[input_df['player_to_predict'] == True]

        # Prepare directory for this week
        week_dir = os.path.join(self.output_dir, f'week_{week}')
        os.makedirs(week_dir, exist_ok=True)

        # Get unique positions
        unique_positions = target_players['player_position'].dropna().unique()

        for pos in unique_positions:
            # Filter distinct sequences for this position
            pos_df = target_players[target_players['player_position'] == pos]
            # Group by sequence
            grouped = pos_df.groupby(['game_id', 'play_id', 'nfl_id'])

            # Get list of sequences
            sequences = list(grouped.groups.keys())

            # Limit
            if limit_per_pos:
                sequences = sequences[:limit_per_pos]

            if not sequences:
                continue

            # Create position subdirectory
            pos_dir = os.path.join(week_dir, pos)
            os.makedirs(pos_dir, exist_ok=True)

            for (game_id, play_id, nfl_id) in sequences:
                try:
                    # 1. Input Data
                    input_seq = grouped.get_group((game_id, play_id, nfl_id))

                    # 2. Output Data (Final Pos)
                    output_seq = output_df[
                        (output_df['game_id'] == game_id) &
                        (output_df['play_id'] == play_id) &
                        (output_df['nfl_id'] == nfl_id)
                    ]

                    if output_seq.empty:
                        continue

                    final_output_pos = output_seq.iloc[-1]

                    # 3. Ball Landing
                    game_row = input_seq.iloc[0]
                    ball_x = game_row.get('ball_land_x', float('nan'))
                    ball_y = game_row.get('ball_land_y', float('nan'))

                    self.plot_sequence(
                        input_seq,
                        final_output_pos,
                        ball_x, ball_y,
                        game_id, play_id, nfl_id,
                        pos,
                        pos_dir
                    )

                except Exception as e:
                    print(f"Error plotting {game_id}-{play_id}-{nfl_id}: {e}")

    def plot_sequence(self, input_df, output_final_row, ball_x, ball_y,
                      game_id, play_id, nfl_id, position, output_dir):
        """Create a plot for the sequence focusing on position context.

        Args:
            input_df: DataFrame with input trajectory data.
            output_final_row: Series with final output position.
            ball_x: Ball landing x-coordinate.
            ball_y: Ball landing y-coordinate.
            game_id: Game identifier.
            play_id: Play identifier.
            nfl_id: Player identifier.
            position: Player position string.
            output_dir: Directory to save the plot.
        """
        plt.figure(figsize=(14, 7))

        # Field Context
        plt.xlim(0, 120)
        plt.ylim(0, 53.3)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.axhline(0, color='black', linewidth=1)
        plt.axhline(53.3, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.axvline(120, color='black', linewidth=1)

        # Yard Lines
        for x in range(10, 110, 10):
            plt.axvline(x, color='gray', linestyle=':', alpha=0.5)

        # 1. Input Trajectory
        plt.plot(
            input_df['x'], input_df['y'],
            'b-', label='Motion Path', alpha=0.6, linewidth=2
        )
        plt.scatter(
            input_df['x'], input_df['y'],
            c=range(len(input_df)), cmap='Blues', s=20, alpha=0.8, edgecolor='none'
        )

        # Start
        plt.plot(
            input_df['x'].iloc[0], input_df['y'].iloc[0],
            'go', label='Start', markersize=8
        )

        # End (Input)
        plt.plot(
            input_df['x'].iloc[-1], input_df['y'].iloc[-1],
            'bs', label='End Input', markersize=8
        )

        # 2. Final Output Position
        plt.plot(
            output_final_row['x'], output_final_row['y'],
            'rx', label='Final Prediction Target', markersize=12, markeredgewidth=3
        )

        # 3. Ball Landing
        if pd.notna(ball_x) and pd.notna(ball_y):
            plt.plot(
                ball_x, ball_y,
                'y*', label='Ball Landing', markersize=15, markeredgecolor='black'
            )

        plt.title(
            f'Position Analysis: {position} | Game {game_id} Play {play_id} '
            f'Player {nfl_id}',
            fontsize=14
        )
        plt.xlabel('Field X (yards)')
        plt.ylabel('Field Y (yards)')
        plt.legend(loc='upper right')

        # Add text box for Role/Position details
        role = input_df['player_role'].iloc[0] if 'player_role' in input_df.columns else 'Unknown'
        info_text = f"Position: {position}\nRole: {role}"
        plt.text(
            2, 51, info_text,
            fontsize=10, bbox=dict(facecolor='white', alpha=0.9, boxstyle='round')
        )

        filename = f"{position}_seq_{game_id}_{play_id}_{nfl_id}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=100)
        plt.close()


def main():
    """Main entry point for the sequence visualizer."""
    # Data directory containing input_*.csv and output_*.csv
    data_dir = (
        '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/'
        'nfl-big-data-bowl-2026-analytics/'
        '114239_nfl_competition_files_published_analytics_final/train'
    )

    # Output directory for plots
    output_dir = (
        '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/'
        'src/visualizations/position_analysis/sequence_by_position'
    )

    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return

    print("Initializing Position-Based Sequence Visualizer...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    try:
        visualizer = SequenceVisualizerByPosition(data_dir, output_dir)

        # Run for all weeks, limited to 5 examples per position per week
        visualizer.generate_all_weeks(limit_per_position=5)

        print(f"\nVisualizations saved to {visualizer.output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
