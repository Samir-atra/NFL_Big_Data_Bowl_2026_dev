
"""
Module for visualizing player movement trajectories on the field.

This tool is essential for visually inspecting the path of all players during a 
specific play, allowing for verification of data quality and intuitive 
understanding of the sequence prediction task, including start/end points 
and key events like ball landing.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_player_trajectories(data_path, output_dir, game_id, play_id):
    """
    Generates a scatter plot of all player trajectories (paths) for a specific 
    play within a given game, indicating start and end points.

    Args:
        data_path (str): The full path to the input CSV file (e.g., a weekly input file).
        output_dir (str): The directory where the resulting plot will be saved.
        game_id (int): The unique identifier for the game.
        play_id (int): The unique identifier for the play within the game.
    """
    # Load the specified input data file
    df = pd.read_csv(data_path)

    # Filter for the specific game and play
    play_df = df[(df['game_id'] == game_id) & (df['play_id'] == play_id)].copy()

    if play_df.empty:
        print(f"No data found for game_id {game_id}, play_id {play_id}")
        return

    plt.figure(figsize=(12, 8))
    
    # Plot player trajectories
    for nfl_id in play_df['nfl_id'].unique():
        player_df = play_df[play_df['nfl_id'] == nfl_id]
        # Plot the entire path
        plt.plot(player_df['x'], player_df['y'], label=f'Player {nfl_id}')
        # Mark the start position (first frame)
        plt.scatter(player_df['x'].iloc[0], player_df['y'].iloc[0], marker='o', s=50, color='green', zorder=5) # Start
        # Mark the end position (last frame)
        plt.scatter(player_df['x'].iloc[-1], player_df['y'].iloc[-1], marker='x', s=50, color='red', zorder=5) # End

    # Plot ball landing position if available in the data
    if 'ball_land_x' in play_df.columns and 'ball_land_y' in play_df.columns:
        # Assuming the ball landing coordinates are constant for all frames/players in a play
        ball_land_x = play_df['ball_land_x'].iloc[0]
        ball_land_y = play_df['ball_land_y'].iloc[0]
        plt.scatter(ball_land_x, ball_land_y, marker='*', s=200, color='orange', label='Ball Landing Spot', zorder=10)

    plt.title(f'Player Trajectories for Game {game_id}, Play {play_id}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # Place legend outside to avoid obscuring the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'player_trajectories_game{game_id}_play{play_id}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved player trajectories plot to {output_path}")

if __name__ == "__main__":
    data_file = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/input_2023_w01.csv"
    output_directory = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/src/visualizations/movement_analysis"
    
    # Example usage: Use the first game_id and play_id from the data
    example_game_id = 2023090700
    example_play_id = 101
    
    plot_player_trajectories(data_file, output_directory, example_game_id, example_play_id)
