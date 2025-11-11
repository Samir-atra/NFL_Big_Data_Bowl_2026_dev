
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_player_trajectories(data_path, output_dir, game_id, play_id):
    """
    Plots the trajectories of all players for a given game and play.
    """
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
        plt.plot(player_df['x'], player_df['y'], label=f'Player {nfl_id}')
        plt.scatter(player_df['x'].iloc[0], player_df['y'].iloc[0], marker='o', s=50, color='green', zorder=5) # Start
        plt.scatter(player_df['x'].iloc[-1], player_df['y'].iloc[-1], marker='x', s=50, color='red', zorder=5) # End

    # Plot ball landing position if available
    if 'ball_land_x' in play_df.columns and 'ball_land_y' in play_df.columns:
        ball_land_x = play_df['ball_land_x'].iloc[0]
        ball_land_y = play_df['ball_land_y'].iloc[0]
        plt.scatter(ball_land_x, ball_land_y, marker='*', s=200, color='orange', label='Ball Landing Spot', zorder=10)

    plt.title(f'Player Trajectories for Game {game_id}, Play {play_id}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'player_trajectories_game{game_id}_play{play_id}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved player trajectories plot to {output_path}")

if __name__ == "__main__":
    data_file = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/input_2023_w01.csv"
    output_directory = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/src/visualizations/movement_analysis"
    
    # Example usage: Use the first game_id and play_id from the data
    # To get these, we might need to read a small portion of the file or assume common ones.
    # From the `head` output, game_id=2023090700, play_id=101 seems to be a good candidate.
    example_game_id = 2023090700
    example_play_id = 101
    
    plot_player_trajectories(data_file, output_directory, example_game_id, example_play_id)
