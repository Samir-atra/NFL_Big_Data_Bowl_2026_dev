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

class OutputDistributionAnalyzer:
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

        print(f"Found {len(input_files)} week files. Aggregating output targets...")
        
        # Structure: {position: {'end_x': [], 'end_y': []}}
        position_targets = {} 

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

                # 2. Read Input Data -> Get POSITION info
                # We only want to map nfl_id/play/game to position
                use_cols = ['game_id', 'play_id', 'nfl_id', 'player_position', 'player_to_predict']
                df_in = pd.read_csv(f, usecols=lambda c: c in use_cols)
                
                # Filter for players to predict
                if 'player_to_predict' in df_in.columns:
                    df_in['player_to_predict'] = df_in['player_to_predict'].astype(str)
                    df_in = df_in[df_in['player_to_predict'] == 'True']
                
                # We just need the unique mapping per play
                df_in = df_in[['game_id', 'play_id', 'nfl_id', 'player_position']].drop_duplicates()

                # 3. Merge to associate Position with Final Output
                merged_df = pd.merge(final_pos, df_in, on=['game_id', 'play_id', 'nfl_id'], how='inner')
                
                # 4. Store by Position
                for pos, group in merged_df.groupby('player_position'):
                    if pos not in position_targets:
                        position_targets[pos] = {'end_x': [], 'end_y': []}
                    
                    position_targets[pos]['end_x'].extend(group['end_x'].tolist())
                    position_targets[pos]['end_y'].extend(group['end_y'].tolist())

            except Exception as e:
                print(f"\nError processing {week_name}: {e}")
        
        print("\nAll weeks processed. Generating output distribution plots...")
        
        # Style settings
        plt.style.use('seaborn-v0_8-darkgrid') 
        
        for pos, data in position_targets.items():
            if not data['end_x']: continue
            
            print(f"Plotting {pos}...", end='\r')
            
            ex = np.array(data['end_x'])
            ey = np.array(data['end_y'])
            
            # Subsample if massive
            count = len(ex)
            if count > 50000:
                indices = np.random.choice(count, 50000, replace=False)
                ex, ey = ex[indices], ey[indices]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.set_facecolor('#f0f0f0') 
            
            # Scatter Plot of Final Destinations
            # Use alpha and distinct color (e.g., Red or heatmap like)
            ax.scatter(ex, ey, c='red', s=15, alpha=0.3, label='Final Output Position')
            
            # KDE plot (optional density - can be slow for many points, stick to scatter or maybe hexbin)
            # ax.hexbin(ex, ey, gridsize=30, cmap='Reds', mincnt=1)
            
            # Field Markings
            ax.axhline(0, color='black', linewidth=1)
            ax.axhline(53.3, color='black', linewidth=1)
            ax.axvline(0, color='black', linewidth=1)
            ax.axvline(120, color='black', linewidth=1)
            for x in range(10, 110, 10):
                ax.axvline(x, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_title(f'Final Output Positions Distribution - Position: {pos}', fontsize=16, pad=15)
            ax.set_xlabel('Field X (yards)', fontsize=12)
            ax.set_ylabel('Field Y (yards)', fontsize=12)
            ax.set_aspect('equal')
            ax.set_xlim(-5, 125)
            ax.set_ylim(-5, 60)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'output_distribution_{pos}.png'), dpi=150)
            plt.close()
            
        print("\nOutput distribution analysis complete.")

if __name__ == "__main__":
    INPUT_DIR = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
    OUTPUT_DIR = "/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-analytics/visualizations/output_distributions"
    
    analyzer = OutputDistributionAnalyzer(INPUT_DIR, OUTPUT_DIR)
    analyzer.analyze_global()
