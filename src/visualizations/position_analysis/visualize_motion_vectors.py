import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

# Ensure backend acts non-interactively
import matplotlib
matplotlib.use('Agg')

class MotionVsFinalAnalyzer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_global(self):
        input_files = glob.glob(os.path.join(self.data_dir, "input_*.csv"))
        input_files.sort()
        
        if not input_files:
            print("No input files found!")
            return

        print(f"Found {len(input_files)} week files. Pairing trajectories...")
        
        # Structure: {position: {'start_x': [], 'start_y': [], 'end_x': [], 'end_y': []}}
        position_vectors = {} 

        for f in input_files:
            week_name = os.path.basename(f).replace('input_', '').replace('.csv', '')
            output_file = os.path.join(self.data_dir, f"output_{week_name}.csv")
            
            if not os.path.exists(output_file):
                continue

            print(f"Processing {week_name}...", end='\r')

            try:
                # 1. Read Output Data -> Get FINAL position per play
                # We assume the last frame in output is the target 'final' position
                df_out = pd.read_csv(output_file)
                # Sort to ensure we get last frame
                df_out = df_out.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
                
                # Get last frame for each player/play
                final_pos = df_out.groupby(['game_id', 'play_id', 'nfl_id'], as_index=False).last()
                final_pos = final_pos.rename(columns={'x': 'end_x', 'y': 'end_y'})
                final_pos = final_pos[['game_id', 'play_id', 'nfl_id', 'end_x', 'end_y']]

                # 2. Read Input Data -> Get LAST KNOWN position (Start of prediction)
                # We only care about player_to_predict=True
                use_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'player_position', 'player_to_predict', 'x', 'y']
                df_in = pd.read_csv(f, usecols=lambda c: c in use_cols)
                
                # Standardize boolean col
                if 'player_to_predict' in df_in.columns:
                    df_in['player_to_predict'] = df_in['player_to_predict'].astype(str)
                    df_in = df_in[df_in['player_to_predict'] == 'True']

                # Get the last frame from input (the 'current' state before prediction)
                df_in = df_in.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
                start_pos = df_in.groupby(['game_id', 'play_id', 'nfl_id'], as_index=False).last()
                start_pos = start_pos.rename(columns={'x': 'start_x', 'y': 'start_y'})
                start_pos = start_pos[['game_id', 'play_id', 'nfl_id', 'player_position', 'start_x', 'start_y']]

                # 3. Merge to create vectors (Start -> End)
                # Inner join ensures we have both start and end for the same play/player
                vectors_df = pd.merge(start_pos, final_pos, on=['game_id', 'play_id', 'nfl_id'], how='inner')
                
                # 4. Store by Position
                for pos, group in vectors_df.groupby('player_position'):
                    if pos not in position_vectors:
                        position_vectors[pos] = {'start_x': [], 'start_y': [], 'end_x': [], 'end_y': []}
                    
                    # Store data
                    # To avoid memory explosion, we can keep subsampling during collection if needed,
                    # but lists of floats are reasonably efficient for ~20 weeks.
                    position_vectors[pos]['start_x'].extend(group['start_x'].tolist())
                    position_vectors[pos]['start_y'].extend(group['start_y'].tolist())
                    position_vectors[pos]['end_x'].extend(group['end_x'].tolist())
                    position_vectors[pos]['end_y'].extend(group['end_y'].tolist())

            except Exception as e:
                print(f"\nError processing {week_name}: {e}")
        
        print("\nAll weeks processed. Generating vector plots...")
        
        plot_dir = self.output_dir
        os.makedirs(plot_dir, exist_ok=True)
        
        # Style settings for "readable" and "premium" look
        plt.style.use('seaborn-v0_8-darkgrid') 
        # Or manually set style
        
        for pos, data in position_vectors.items():
            if not data['start_x']: continue
            
            print(f"Plotting {pos}...", end='\r')
            
            # Convert to numpy for easier handling
            sx = np.array(data['start_x'])
            sy = np.array(data['start_y'])
            ex = np.array(data['end_x'])
            ey = np.array(data['end_y'])
            
            count = len(sx)
            
            # Smart Subsampling
            # If too many points, randomly sample N points to keep plot readable
            MAX_ARROWS = 800
            if count > MAX_ARROWS:
                indices = np.random.choice(count, MAX_ARROWS, replace=False)
                sx, sy, ex, ey = sx[indices], sy[indices], ex[indices], ey[indices]
            
            # Calculate vectors
            u = ex - sx
            v = ey - sy
            mag = np.sqrt(u**2 + v**2) # Distance traveled
            
            # Create Plot
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot Field Background Context (optional, but helps readability)
            ax.set_facecolor('#f0f0f0') # Light gray background
            
            # Quiver Plot (Arrows)
            # Color by Magnitude (Distance) to show "explosiveness" or "long runs"
            # Cmap: 'plasma' (purple-orange-yellow) stands out well on light bg
            q = ax.quiver(sx, sy, u, v, mag, 
                          angles='xy', scale_units='xy', scale=1, 
                          cmap='plasma', alpha=0.8, width=0.003, headwidth=4, headlength=5)
            
            # Add Scatter for endpoints to mark the final spot firmly
            # ax.scatter(ex, ey, c='black', s=10, alpha=0.5, marker='.', zorder=2)
            
            # Colorbar
            cbar = plt.colorbar(q, ax=ax)
            cbar.set_label('Distance Traveled (yards)')
            
            # Field Markings
            ax.axhline(0, color='black', linewidth=1)
            ax.axhline(53.3, color='black', linewidth=1)
            ax.axvline(0, color='black', linewidth=1)
            ax.axvline(120, color='black', linewidth=1)
            # Yard lines
            for x in range(10, 110, 10):
                ax.axvline(x, color='gray', linestyle=':', alpha=0.5)
            
            # Labels
            ax.set_title(f'Player Motion Vectors: Start to Final Position - Position: {pos}', fontsize=16, pad=15)
            ax.set_xlabel('Field X (yards)', fontsize=12)
            ax.set_ylabel('Field Y (yards)', fontsize=12)
            
            # Fix aspect ratio to represent real field
            ax.set_aspect('equal')
            
            # Interactive-like limits
            ax.set_xlim(-5, 125)
            ax.set_ylim(-5, 60)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'vector_motion_{pos}.png'), dpi=150)
            plt.close()
            
        print("\nVector analysis complete.")

if __name__ == "__main__":
    INPUT_DIR = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
    OUTPUT_DIR = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/visualizations/motion_vectors"
    
    analyzer = MotionVsFinalAnalyzer(INPUT_DIR, OUTPUT_DIR)
    analyzer.analyze_global()
