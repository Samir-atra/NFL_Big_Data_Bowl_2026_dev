
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_player_speed_acceleration(data_path, output_dir, game_id, play_id):
    """
    Plots the speed and acceleration of players over time for a given game and play.
    """
    df = pd.read_csv(data_path)

    # Filter for the specific game and play
    play_df = df[(df['game_id'] == game_id) & (df['play_id'] == play_id)].copy()

    if play_df.empty:
        print(f"No data found for game_id {game_id}, play_id {play_id}")
        return

    unique_nfl_ids = play_df['nfl_id'].unique()
    
    # Plotting Speed
    plt.figure(figsize=(14, 7))
    for nfl_id in unique_nfl_ids:
        player_df = play_df[play_df['nfl_id'] == nfl_id]
        plt.plot(player_df['frame_id'], player_df['s'], label=f'Player {nfl_id}')
    plt.title(f'Player Speed Over Time for Game {game_id}, Play {play_id}')
    plt.xlabel('Frame ID')
    plt.ylabel('Speed (s)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    output_path_speed = os.path.join(output_dir, f'player_speed_game{game_id}_play{play_id}.png')
    plt.savefig(output_path_speed)
    plt.close()
    print(f"Saved player speed plot to {output_path_speed}")

    # Plotting Acceleration
    plt.figure(figsize=(14, 7))
    for nfl_id in unique_nfl_ids:
        player_df = play_df[play_df['nfl_id'] == nfl_id]
        plt.plot(player_df['frame_id'], player_df['a'], label=f'Player {nfl_id}')
    plt.title(f'Player Acceleration Over Time for Game {game_id}, Play {play_id}')
    plt.xlabel('Frame ID')
    plt.ylabel('Acceleration (a)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    output_path_accel = os.path.join(output_dir, f'player_acceleration_game{game_id}_play{play_id}.png')
    plt.savefig(output_path_accel)
    plt.close()
    print(f"Saved player acceleration plot to {output_path_accel}")

if __name__ == "__main__":
    data_file = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/input_2023_w01.csv"
    output_directory = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/src/visualizations/movement_analysis"
    
    example_game_id = 2023090700
    example_play_id = 101
    
    plot_player_speed_acceleration(data_file, output_directory, example_game_id, example_play_id)
