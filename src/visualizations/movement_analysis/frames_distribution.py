
"""
Module for analyzing and visualizing the distribution of sequence lengths (frames)
in the NFL Big Data Bowl datasets.

The primary function `plot_frames_distribution` helps in understanding the variance
in the duration of plays, which is crucial for determining the maximum sequence 
length for fixed-length sequence models.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_frames_distribution(data_path, output_dir):
    """
    Calculates the number of frames for each unique play and generates a histogram
    to visualize the distribution of sequence lengths.

    Args:
        data_path (str): The full path to the input CSV file (e.g., a weekly input file).
        output_dir (str): The directory where the resulting plot will be saved.
    """
    # Load the specified input data file
    df = pd.read_csv(data_path)

    # Group by play IDs and find the maximum frame_id, which represents the total frames in that play
    frames_per_play = df.groupby(['game_id', 'play_id'])['frame_id'].max()

    # Check if any data was processed
    if frames_per_play.empty:
        print(f"No play data found in {data_path}")
        return

    # Create and configure the histogram plot
    plt.figure(figsize=(10, 6))
    frames_per_play.hist(bins=50)
    plt.title('Distribution of Frames per Play')
    plt.xlabel('Number of Frames')
    plt.ylabel('Number of Plays')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to the specified output directory
    output_path = os.path.join(output_dir, 'frames_per_play_distribution.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved frames per play distribution plot to {output_path}")

if __name__ == "__main__":
    data_file = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train/input_2023_w01.csv"
    output_directory = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/src/visualizations/movement_analysis"
    
    # Run the visualization for a sample week
    plot_frames_distribution(data_file, output_directory)
