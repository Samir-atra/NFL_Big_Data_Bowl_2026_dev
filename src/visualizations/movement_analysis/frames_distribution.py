
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_frames_distribution(data_path, output_dir):
    """
    Plots the distribution of the number of frames per play.
    """
    df = pd.read_csv(data_path)

    # Calculate the number of frames per play
    frames_per_play = df.groupby(['game_id', 'play_id'])['frame_id'].max()

    if frames_per_play.empty:
        print(f"No play data found in {data_path}")
        return

    plt.figure(figsize=(10, 6))
    frames_per_play.hist(bins=50)
    plt.title('Distribution of Frames per Play')
    plt.xlabel('Number of Frames')
    plt.ylabel('Number of Plays')
    plt.grid(True)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'frames_per_play_distribution.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved frames per play distribution plot to {output_path}")

if __name__ == "__main__":
    data_file = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/input_2023_w01.csv"
    output_directory = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/src/visualizations/movement_analysis"
    
    plot_frames_distribution(data_file, output_directory)
